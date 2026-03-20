"""VSFA — Quality-Aware Features for Video Quality Assessment.

Li et al. ACMMM 2019 — uses pre-trained deep CNN features with a GRU
for temporal quality-aware aggregation. Learns to weight frame-level
features by their quality relevance.

GitHub: https://github.com/lidq92/VSFA

vsfa_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _spatial_quality_features(gray: np.ndarray) -> np.ndarray:
    """Extract spatial quality features as CNN proxy."""
    h, w = gray.shape

    # Sharpness (Laplacian variance)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Multi-scale contrast
    contrasts = []
    current = gray.copy()
    for _ in range(3):
        contrasts.append(current.std())
        if current.shape[0] > 16 and current.shape[1] > 16:
            current = cv2.pyrDown(current)

    # Edge density
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edge_density = np.mean(edges > 0)

    # Texture energy via Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    texture_energy = np.mean(np.sqrt(sobel_x ** 2 + sobel_y ** 2))

    return np.array([
        lap_var,
        *contrasts,
        edge_density,
        texture_energy,
    ], dtype=np.float64)


class VSFAModule(PipelineModule):
    name = "vsfa"
    description = "VSFA quality-aware feature aggregation with GRU (ACMMM 2019)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: Try native VSFA
        try:
            import vsfa
            self._model = vsfa
            self._backend = "native"
            logger.info("VSFA (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: Heuristic fallback
        self._backend = "heuristic"
        logger.info("VSFA (heuristic) initialised — install vsfa for full model")

    def process(self, sample: Sample) -> Sample:
        try:
            if self._backend == "native":
                score = float(self._model.predict(str(sample.path)))
            else:
                score = self._process_heuristic(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.vsfa_score = score

        except Exception as e:
            logger.warning(f"VSFA failed for {sample.path}: {e}")

        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: spatial features + GRU-like temporal aggregation."""
        frames_gray = []

        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return None
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                        frames_gray.append(gray)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is None:
                return None
            frames_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64))

        if not frames_gray:
            return None

        # Extract per-frame spatial features
        frame_features = []
        for gray in frames_gray:
            frame_features.append(_spatial_quality_features(gray))

        feat_matrix = np.array(frame_features)

        # Per-frame quality scores (simple spatial quality)
        frame_scores = []
        for feats in frame_features:
            sharpness = min(feats[0] / 800.0, 1.0)
            contrast = min(feats[1] / 70.0, 1.0)
            edge_density = min(feats[4] / 0.15, 1.0)
            texture = min(feats[5] / 40.0, 1.0)
            q = 0.35 * sharpness + 0.25 * contrast + 0.20 * edge_density + 0.20 * texture
            frame_scores.append(q)

        frame_scores = np.array(frame_scores)

        # GRU-like temporal aggregation: quality-aware weighting
        # Frames with higher quality features get more weight (attention proxy)
        if len(frame_scores) > 1:
            # Temporal consistency bonus
            temporal_diff = np.abs(np.diff(frame_scores))
            consistency = 1.0 / (1.0 + np.mean(temporal_diff) * 5.0)

            # Weighted mean: use softmax-like weighting on frame quality
            weights = np.exp(frame_scores * 2.0)
            weights = weights / (np.sum(weights) + 1e-7)
            weighted_score = np.sum(weights * frame_scores)

            score = 0.8 * weighted_score + 0.2 * consistency
        else:
            score = float(frame_scores[0])

        return float(np.clip(score, 0.0, 1.0))

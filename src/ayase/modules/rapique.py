"""RAPIQUE — Rapid and Accurate Video Quality Prediction of UGC.

IEEE OJSP 2021 — combines bandpass natural scene statistics (NSS)
with deep CNN semantic features for space-time quality prediction.
Orders-of-magnitude faster than SOTA with comparable accuracy.

GitHub: https://github.com/vztu/RAPIQUE

rapique_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _compute_nss_features(gray: np.ndarray) -> np.ndarray:
    """Compute bandpass NSS features (simplified BRISQUE-like)."""
    h, w = gray.shape
    mu = cv2.GaussianBlur(gray, (7, 7), 7 / 6)
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(gray * gray, (7, 7), 7 / 6) - mu_sq))
    sigma = np.maximum(sigma, 1e-7)
    mscn = (gray - mu) / sigma

    features = []
    features.append(np.mean(mscn))
    features.append(np.var(mscn))
    features.append(float(np.mean(np.abs(mscn))))

    # Paired product features (horizontal, vertical)
    for shift in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        shifted = np.roll(np.roll(mscn, shift[0], axis=0), shift[1], axis=1)
        paired = mscn * shifted
        features.append(np.mean(paired))
        features.append(np.var(paired))

    return np.array(features, dtype=np.float32)


def _compute_spatial_features(frame: np.ndarray) -> np.ndarray:
    """Compute spatial quality features."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Sharpness via Laplacian
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Contrast
    contrast = gray.std()

    # Colorfulness
    b, g, r = frame[:, :, 0].astype(float), frame[:, :, 1].astype(float), frame[:, :, 2].astype(float)
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)

    # NSS features
    nss = _compute_nss_features(gray)

    return np.concatenate([[lap_var, contrast, colorfulness], nss])


class RAPIQUEModule(PipelineModule):
    name = "rapique"
    description = "RAPIQUE rapid NR-VQA via bandpass NSS + CNN features (IEEE OJSP 2021)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._ml_available = False
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: Try native RAPIQUE
        try:
            import rapique
            self._model = rapique
            self._ml_available = True
            self._backend = "native"
            logger.info("RAPIQUE (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: Heuristic fallback always available
        self._backend = "heuristic"
        logger.info("RAPIQUE (heuristic) initialised — install rapique for full model")

    def process(self, sample: Sample) -> Sample:
        try:
            if self._backend == "native":
                score = self._process_native(sample)
            else:
                score = self._process_heuristic(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.rapique_score = score

        except Exception as e:
            logger.warning(f"RAPIQUE failed for {sample.path}: {e}")

        return sample

    def _process_native(self, sample: Sample) -> Optional[float]:
        score = self._model.predict(str(sample.path))
        return float(score)

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: bandpass NSS + spatial features, SVR-like combination."""
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return None

            indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
            all_features = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    all_features.append(_compute_spatial_features(frame))
            cap.release()

            if not all_features:
                return None

            features = np.mean(all_features, axis=0)

            # Temporal features: variance across frames
            feat_stack = np.array(all_features)
            temporal_var = np.mean(np.var(feat_stack, axis=0))
        else:
            img = cv2.imread(str(sample.path))
            if img is None:
                return None
            features = _compute_spatial_features(img)
            temporal_var = 0.0

        # Heuristic quality mapping
        sharpness = min(features[0] / 1000.0, 1.0)
        contrast = min(features[1] / 80.0, 1.0)
        colorfulness = min(features[2] / 120.0, 1.0)
        nss_regularity = 1.0 / (1.0 + abs(features[3]))
        temporal_stability = 1.0 / (1.0 + temporal_var * 0.01)

        score = (
            0.30 * sharpness
            + 0.20 * contrast
            + 0.15 * colorfulness
            + 0.20 * nss_regularity
            + 0.15 * temporal_stability
        )

        return float(np.clip(score, 0.0, 1.0))

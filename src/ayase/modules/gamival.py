"""GAMIVAL — Gaming Video Quality Assessment.

Yu et al. 2023 — NR-VQA specifically designed for cloud gaming content.
Combines NSS features with CNN features tailored for gaming artifacts
like encoding distortions, frame drops, and rendering issues.

gamival_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _gaming_artifact_features(frame: np.ndarray) -> np.ndarray:
    """Extract features relevant to gaming content quality."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape

    # Sharpness (gaming content should be sharp)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Blockiness detection (common in compressed gaming streams)
    block_size = 8
    h_blocks = h // block_size
    w_blocks = w // block_size
    blockiness = 0.0
    if h_blocks > 1 and w_blocks > 1:
        cropped = gray[: h_blocks * block_size, : w_blocks * block_size]
        # Measure discontinuities across block boundaries
        h_edges = np.abs(cropped[block_size-1::block_size, :].astype(float) - cropped[block_size::block_size, :].astype(float))
        v_edges = np.abs(cropped[:, block_size-1::block_size].astype(float) - cropped[:, block_size::block_size].astype(float))
        blockiness = float(np.mean(h_edges) + np.mean(v_edges))

    # Banding detection (gradient posterization common in gaming)
    gradient_y = np.abs(np.diff(gray, axis=0))
    gradient_x = np.abs(np.diff(gray, axis=1))
    # Low gradient areas that should be smooth
    smooth_mask_y = gradient_y < 2.0
    smooth_mask_x = gradient_x < 2.0
    banding = float(np.mean(smooth_mask_y) + np.mean(smooth_mask_x)) / 2.0

    # Colorfulness (gaming often has vivid colors)
    b, g, r = (
        frame[:, :, 0].astype(np.float64),
        frame[:, :, 1].astype(np.float64),
        frame[:, :, 2].astype(np.float64),
    )
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(
        rg.mean() ** 2 + yb.mean() ** 2
    )

    # Contrast
    contrast = gray.std()

    # NSS features (MSCN)
    mu = cv2.GaussianBlur(gray, (7, 7), 7 / 6)
    sigma = np.sqrt(
        np.abs(cv2.GaussianBlur(gray * gray, (7, 7), 7 / 6) - mu * mu)
    )
    sigma = np.maximum(sigma, 1e-7)
    mscn = (gray - mu) / sigma
    nss_mean = float(np.mean(np.abs(mscn)))
    nss_var = float(np.var(mscn))

    return np.array([
        lap_var,        # sharpness
        blockiness,     # block artifacts
        banding,        # gradient banding
        colorfulness,   # color saturation
        contrast,       # luminance contrast
        nss_mean,       # NSS regularity
        nss_var,        # NSS variance
    ], dtype=np.float64)


class GAMIVALModule(PipelineModule):
    name = "gamival"
    description = "GAMIVAL cloud gaming NR-VQA with NSS + CNN features (2023)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: Try native GAMIVAL
        try:
            import gamival
            self._model = gamival
            self._backend = "native"
            logger.info("GAMIVAL (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: Heuristic fallback
        self._backend = "heuristic"
        logger.info("GAMIVAL (heuristic) initialised — install gamival for full model")

    def process(self, sample: Sample) -> Sample:
        try:
            if self._backend == "native":
                score = float(self._model.predict(str(sample.path)))
            else:
                score = self._process_heuristic(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.gamival_score = score

        except Exception as e:
            logger.warning(f"GAMIVAL failed for {sample.path}: {e}")

        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: gaming-specific artifact detection + NSS features."""
        frames = []

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
                        frames.append(frame)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)

        if not frames:
            return None

        # Extract gaming-specific features per frame
        all_features = []
        for frame in frames:
            all_features.append(_gaming_artifact_features(frame))

        feat_matrix = np.array(all_features)
        feat_mean = np.mean(feat_matrix, axis=0)

        # Quality components
        sharpness = min(feat_mean[0] / 800.0, 1.0)
        # Low blockiness is good
        block_quality = 1.0 / (1.0 + feat_mean[1] * 0.05)
        # Low banding is good (banding ~0.5 means half pixels have near-zero gradient)
        band_quality = 1.0 - min(feat_mean[2], 1.0) * 0.5
        colorfulness = min(feat_mean[3] / 100.0, 1.0)
        contrast = min(feat_mean[4] / 65.0, 1.0)
        nss_regularity = 1.0 / (1.0 + abs(feat_mean[5] - 0.8))

        spatial_score = (
            0.25 * sharpness
            + 0.20 * block_quality
            + 0.10 * band_quality
            + 0.15 * colorfulness
            + 0.15 * contrast
            + 0.15 * nss_regularity
        )

        # Temporal quality (for video)
        if len(frames) > 1:
            # Frame drop detection: large sudden changes
            grays = [
                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64) for f in frames
            ]
            diffs = []
            for i in range(len(grays) - 1):
                diffs.append(np.mean(np.abs(grays[i + 1] - grays[i])))
            diffs = np.array(diffs)
            # Consistency of motion (frame drops cause spikes)
            temporal = 1.0 / (1.0 + np.std(diffs) * 0.1)
            score = 0.75 * spatial_score + 0.25 * temporal
        else:
            score = spatial_score

        return float(np.clip(score, 0.0, 1.0))

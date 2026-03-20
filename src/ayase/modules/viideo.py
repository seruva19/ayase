"""VIIDEO — Video Intrinsic Integrity and Distortion Evaluation Oracle.

Mittal et al. 2016 — completely blind NR-VQA using natural video
statistics of frame differences. No training data or human opinions
required; relies on statistical regularity of natural videos.

viideo_score — LOWER = better quality (distortion measure)
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _compute_nss_params(patch: np.ndarray) -> tuple:
    """Compute generalized Gaussian distribution parameters for NSS."""
    mu = np.mean(patch)
    sigma = np.std(patch) + 1e-7
    skew = float(np.mean(((patch - mu) / sigma) ** 3))
    kurt = float(np.mean(((patch - mu) / sigma) ** 4))
    return mu, sigma, skew, kurt


def _frame_difference_features(gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
    """Compute NSS features on frame difference."""
    diff = gray2.astype(np.float64) - gray1.astype(np.float64)

    # MSCN of frame difference
    mu = cv2.GaussianBlur(diff, (7, 7), 7 / 6)
    sigma = np.sqrt(
        np.abs(cv2.GaussianBlur(diff * diff, (7, 7), 7 / 6) - mu * mu)
    )
    sigma = np.maximum(sigma, 1e-7)
    mscn = (diff - mu) / sigma

    features = []

    # Global statistics of MSCN coefficients
    features.append(np.mean(mscn))
    features.append(np.var(mscn))
    features.append(float(np.mean(np.abs(mscn))))

    # Kurtosis
    std_val = np.std(mscn) + 1e-7
    features.append(float(np.mean(((mscn - np.mean(mscn)) / std_val) ** 4)))

    # Paired product features (4 orientations)
    for shift in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        shifted = np.roll(np.roll(mscn, shift[0], axis=0), shift[1], axis=1)
        paired = mscn * shifted
        features.append(np.mean(paired))
        features.append(np.var(paired))

    # Energy of difference
    features.append(np.mean(diff ** 2))

    return np.array(features, dtype=np.float64)


class VIIDEOModule(PipelineModule):
    name = "viideo"
    description = "VIIDEO blind NR-VQA via natural video statistics (Mittal 2016, lower=better)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: Try native VIIDEO package
        try:
            import viideo
            self._model = viideo
            self._backend = "native"
            logger.info("VIIDEO (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: Heuristic fallback always available
        self._backend = "heuristic"
        logger.info("VIIDEO (heuristic) initialised — install viideo for full model")

    def process(self, sample: Sample) -> Sample:
        try:
            if self._backend == "native":
                score = float(self._model.predict(str(sample.path)))
            else:
                score = self._process_heuristic(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.viideo_score = score

        except Exception as e:
            logger.warning(f"VIIDEO failed for {sample.path}: {e}")

        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: NSS features on frame differences — lower = better."""
        if not sample.is_video:
            # VIIDEO is video-only; for images return neutral score
            return 0.0

        cap = cv2.VideoCapture(str(sample.path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 1:
                return None

            n_frames = min(self.subsample + 1, total)
            indices = np.linspace(0, total - 1, n_frames, dtype=int)

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                    frames.append(gray)
        finally:
            cap.release()

        if len(frames) < 2:
            return None

        # Compute NSS features on consecutive frame differences
        all_features = []
        for i in range(len(frames) - 1):
            feats = _frame_difference_features(frames[i], frames[i + 1])
            all_features.append(feats)

        feat_matrix = np.array(all_features)

        # VIIDEO intrinsic measure: higher variance in NSS params = more distortion
        # Across-time variance of features indicates temporal irregularity
        temporal_var = np.mean(np.var(feat_matrix, axis=0))

        # Mean energy of frame differences (motion-compensated distortion)
        mean_energy = np.mean(feat_matrix[:, -1])  # last feature is diff energy

        # NSS regularity: deviation from expected statistics
        mean_abs_mscn = np.mean(feat_matrix[:, 2])  # mean abs MSCN
        nss_deviation = abs(mean_abs_mscn - 0.798)  # ideal GGD value ~0.798

        # Combine into distortion score (lower = better quality)
        score = (
            0.35 * min(temporal_var * 0.1, 1.0)
            + 0.30 * min(mean_energy / 2000.0, 1.0)
            + 0.35 * min(nss_deviation * 2.0, 1.0)
        )

        return float(np.clip(score, 0.0, 1.0))

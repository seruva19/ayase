"""VIIDEO — Video Intrinsic Integrity and Distortion Evaluation Oracle.

Mittal et al. 2016 — completely blind NR-VQA using natural scene
statistics (NSS) of frame differences. No training data or human
opinions required; relies on statistical regularity of natural videos.

Algorithm:
  1. Compute frame differences between adjacent frames.
  2. Apply MSCN (mean subtracted contrast normalisation) to each
     frame difference.
  3. Fit Asymmetric Generalized Gaussian Distribution (AGGD) to the
     MSCN coefficients.
  4. Compute pairwise statistics of adjacent frame-difference MSCN maps.
  5. Distortion = deviation from expected natural video statistics.

Backend tiers:
  1. scikit-video (skvideo.measure.viideo_score) — canonical implementation
  2. Built-in MSCN + AGGD on frame differences (paper's algorithm)

viideo_score — LOWER = better quality (distortion measure)
"""

import logging
import cv2
import numpy as np
from typing import Optional, List

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AGGD fitting (Asymmetric Generalized Gaussian Distribution)
# ---------------------------------------------------------------------------

def _estimate_aggd_params(data: np.ndarray) -> tuple:
    """Fit AGGD to data via moment matching.

    Returns (beta, sigma_left, sigma_right, mean).
    AGGD allows asymmetric tails, which is characteristic of
    paired MSCN products in natural images/video.
    """
    data = data.flatten().astype(np.float64)
    if len(data) < 10:
        return 2.0, 1.0, 1.0, 0.0

    left = data[data < 0]
    right = data[data >= 0]

    sigma_l = np.sqrt(np.mean(left ** 2)) + 1e-10 if len(left) > 1 else 1e-10
    sigma_r = np.sqrt(np.mean(right ** 2)) + 1e-10 if len(right) > 1 else 1e-10

    # Estimate shape via moment ratio
    sigma = np.std(data) + 1e-10
    mean_abs = np.mean(np.abs(data - np.mean(data))) + 1e-10
    r = sigma / mean_abs
    if r < 1.05:
        beta = max(10.0, 50.0 * (1.05 - r))
    elif r < 1.2533:
        beta = 2.0 + (1.2533 - r) / (1.2533 - 1.05) * 8.0
    elif r < 1.4142:
        beta = 1.0 + (1.4142 - r) / (1.4142 - 1.2533) * 1.0
    else:
        beta = max(0.2, 1.0 / (r - 0.9 + 1e-10))

    return float(beta), float(sigma_l), float(sigma_r), float(np.mean(data))


def _estimate_ggd_params(data: np.ndarray) -> tuple:
    """Fit GGD to data via moment matching. Returns (beta, sigma)."""
    data = data.flatten().astype(np.float64)
    if len(data) < 10:
        return 2.0, 1.0
    sigma = np.std(data) + 1e-10
    mean_abs = np.mean(np.abs(data - np.mean(data))) + 1e-10
    r = sigma / mean_abs
    if r < 1.05:
        beta = max(10.0, 50.0 * (1.05 - r))
    elif r < 1.2533:
        beta = 2.0 + (1.2533 - r) / (1.2533 - 1.05) * 8.0
    elif r < 1.4142:
        beta = 1.0 + (1.4142 - r) / (1.4142 - 1.2533) * 1.0
    else:
        beta = max(0.2, 1.0 / (r - 0.9 + 1e-10))
    return float(beta), float(sigma)


# ---------------------------------------------------------------------------
# Frame-difference NSS features
# ---------------------------------------------------------------------------

def _compute_mscn(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Compute MSCN (Mean Subtracted Contrast Normalised) coefficients."""
    mu = cv2.GaussianBlur(image, (kernel_size, kernel_size), kernel_size / 6)
    sigma = np.sqrt(
        np.abs(cv2.GaussianBlur(image ** 2, (kernel_size, kernel_size), kernel_size / 6) - mu ** 2)
    )
    sigma = np.maximum(sigma, 1e-7)
    return (image - mu) / sigma


def _frame_difference_nss(gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
    """Compute NSS features on a single frame difference.

    VIIDEO core: MSCN on frame difference, then AGGD fitting and
    pairwise product statistics.

    Returns 18 features per frame-difference pair.
    """
    diff = gray2.astype(np.float64) - gray1.astype(np.float64)

    # MSCN of frame difference
    mscn = _compute_mscn(diff)

    features = []

    # GGD parameters of MSCN coefficients (2)
    beta, sigma = _estimate_ggd_params(mscn)
    features.extend([beta, sigma])

    # MSCN statistics (4)
    mscn_flat = mscn.flatten()
    m = np.mean(mscn_flat)
    s = np.std(mscn_flat) + 1e-10
    skew = float(np.mean(((mscn_flat - m) / s) ** 3))
    kurt = float(np.mean(((mscn_flat - m) / s) ** 4) - 3.0)
    features.extend([m, float(np.var(mscn_flat)), skew, kurt])

    # AGGD fit on paired products in 4 orientations (4 x 3 = 12)
    shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dy, dx in shifts:
        if dy == 0:
            paired = mscn[:, :-abs(dx)] * mscn[:, abs(dx):]
        elif dx == 0:
            paired = mscn[:-abs(dy), :] * mscn[abs(dy):, :]
        elif dx > 0:
            paired = mscn[:-dy, :-dx] * mscn[dy:, dx:]
        else:
            paired = mscn[:-dy, -dx:] * mscn[dy:, :dx]

        aggd_beta, aggd_sl, aggd_sr, aggd_mean = _estimate_aggd_params(paired)
        features.extend([aggd_beta, aggd_sl, aggd_sr])

    return np.array(features[:18], dtype=np.float64)


def _pairwise_frame_diff_features(
    feats_a: np.ndarray, feats_b: np.ndarray
) -> np.ndarray:
    """Pairwise statistics between adjacent frame-difference features.

    VIIDEO computes correlations/differences between NSS parameters
    of adjacent frame differences to detect temporal irregularity.
    Returns 6 features.
    """
    diff = feats_a - feats_b
    return np.array([
        np.mean(diff),
        np.std(diff),
        np.mean(np.abs(diff)),
        np.max(np.abs(diff)),
        # Correlation
        float(np.corrcoef(feats_a, feats_b)[0, 1])
        if np.std(feats_a) > 1e-8 and np.std(feats_b) > 1e-8
        else 0.0,
        # Ratio of norms
        float(np.linalg.norm(feats_a) / (np.linalg.norm(feats_b) + 1e-10)),
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class VIIDEOModule(PipelineModule):
    name = "viideo"
    description = "VIIDEO blind NR-VQA via natural video statistics (Mittal 2016, lower=better)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._skvideo_fn = None
        self._ml_available = True
        self._backend = "builtin"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: scikit-video canonical implementation
        try:
            from skvideo.measure import viideo_score as skvideo_viideo
            self._skvideo_fn = skvideo_viideo
            self._backend = "skvideo"
            logger.info("VIIDEO initialised (scikit-video backend)")
            return
        except ImportError:
            pass

        # Tier 2: built-in MSCN + AGGD implementation
        self._backend = "builtin"
        logger.info(
            "VIIDEO initialised (built-in AGGD). "
            "Install scikit-video for canonical implementation: pip install scikit-video"
        )

    def process(self, sample: Sample) -> Sample:
        try:
            if self._backend == "skvideo":
                score = self._process_skvideo(sample)
            else:
                score = self._process_builtin(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.viideo_score = score

        except Exception as e:
            logger.warning("VIIDEO failed for %s: %s", sample.path, e)

        return sample

    def _process_skvideo(self, sample: Sample) -> Optional[float]:
        """Use scikit-video's viideo_score() as primary backend."""
        try:
            import skvideo.io

            if not sample.is_video:
                return 0.0  # VIIDEO is video-only

            video_data = skvideo.io.vread(str(sample.path))
            score = self._skvideo_fn(video_data)

            if isinstance(score, np.ndarray):
                score = float(score.flatten()[0])

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            logger.warning("VIIDEO skvideo failed, falling back to built-in: %s", e)
            return self._process_builtin(sample)

    def _process_builtin(self, sample: Sample) -> Optional[float]:
        """Built-in: MSCN + AGGD on frame differences (lower = better)."""
        if not sample.is_video:
            return 0.0  # VIIDEO is video-only; neutral score for images

        frames_gray = self._load_frames_gray(sample)
        if len(frames_gray) < 2:
            return None

        # --- Step 1: Compute NSS features on each frame difference ---
        frame_diff_features = []
        for i in range(len(frames_gray) - 1):
            feats = _frame_difference_nss(frames_gray[i], frames_gray[i + 1])
            frame_diff_features.append(feats)

        if len(frame_diff_features) < 1:
            return None

        feat_matrix = np.array(frame_diff_features)  # (N_diffs, 18)

        # --- Step 2: Pairwise statistics of adjacent frame-difference features ---
        pairwise_features = []
        for i in range(len(frame_diff_features) - 1):
            pw = _pairwise_frame_diff_features(
                frame_diff_features[i], frame_diff_features[i + 1]
            )
            pairwise_features.append(pw)

        # --- Step 3: Compute distortion score ---
        # VIIDEO intrinsic measure: deviation of NSS parameters from
        # expected natural video statistics indicates distortion.

        # Across-time variance of features: higher = more temporal irregularity
        temporal_variance = np.mean(np.var(feat_matrix, axis=0))

        # GGD shape deviation from expected natural value (~2.0 for Gaussian)
        mean_shape = np.mean(feat_matrix[:, 0])
        shape_deviation = abs(mean_shape - 2.0)

        # AGGD asymmetry: natural videos have roughly symmetric paired products
        # AGGD sigma_left vs sigma_right for each paired product
        aggd_asymmetries = []
        for pair_idx in range(4):
            base = 6 + pair_idx * 3  # indices for AGGD params
            if base + 2 < feat_matrix.shape[1]:
                sl = feat_matrix[:, base + 1]  # sigma_left
                sr = feat_matrix[:, base + 2]  # sigma_right
                asymmetry = np.mean(np.abs(sl - sr) / (sl + sr + 1e-10))
                aggd_asymmetries.append(asymmetry)
        mean_asymmetry = np.mean(aggd_asymmetries) if aggd_asymmetries else 0.0

        # Pairwise feature consistency
        if pairwise_features:
            pw_matrix = np.array(pairwise_features)
            # Mean absolute difference between adjacent frame-difference features
            temporal_irregularity = np.mean(pw_matrix[:, 2])  # mean abs diff
            # Low correlation between adjacent = irregular
            mean_pw_corr = np.mean(pw_matrix[:, 4])
            correlation_deviation = max(0.0, 1.0 - mean_pw_corr)
        else:
            temporal_irregularity = 0.0
            correlation_deviation = 0.0

        # Combine into final distortion score (lower = better quality)
        score = (
            0.25 * min(temporal_variance * 0.1, 1.0)
            + 0.20 * min(shape_deviation * 0.5, 1.0)
            + 0.20 * min(mean_asymmetry * 2.0, 1.0)
            + 0.15 * min(temporal_irregularity * 0.5, 1.0)
            + 0.20 * min(correlation_deviation, 1.0)
        )

        return float(np.clip(score, 0.0, 1.0))

    def _load_frames_gray(self, sample: Sample) -> List[np.ndarray]:
        """Load frames as float64 grayscale arrays."""
        cap = cv2.VideoCapture(str(sample.path))
        frames = []
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 1:
                return frames

            # Need subsample+1 frames to get subsample frame differences
            n_frames = min(self.subsample + 1, total)
            indices = np.linspace(0, total - 1, n_frames, dtype=int)

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                    frames.append(gray)
        finally:
            cap.release()

        return frames

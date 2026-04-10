"""V-BLIINDS — Video BLind Image Integrity Notator using DCT Statistics.

Saad & Bovik, 2014 — blind NR-VQA using DCT-domain natural scene
statistics (NSS) and motion coherency features from optical flow.

Algorithm:
  1. DCT coefficients of local patches -> fit Generalized Gaussian
     Distribution (GGD) parameters (shape, sigma) per frame.
  2. Motion coherency features via optical flow correlation.
  3. ~46 features total -> SVR for quality prediction.

Backend tiers:
  1. scikit-video (skvideo.measure.videobliinds_features) — canonical impl
  2. Built-in DCT + GGD + motion coherency (paper's algorithm)

vbliinds_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional, List

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GGD fitting
# ---------------------------------------------------------------------------

def _estimate_ggd_params(data: np.ndarray) -> tuple:
    """Estimate Generalized Gaussian Distribution shape and sigma.

    Uses the moment-ratio method: ratio of standard deviation to mean
    absolute deviation estimates the shape parameter beta.

    Returns (beta, sigma).
    """
    data = data.flatten().astype(np.float64)
    if len(data) < 10:
        return 2.0, 1.0

    sigma = np.std(data) + 1e-10
    mean_abs = np.mean(np.abs(data - np.mean(data))) + 1e-10

    # r = sigma / mean_abs_dev
    # For Gaussian (beta=2): r = sqrt(pi/2) ~ 1.2533
    # For Laplacian (beta=1): r = sqrt(2) ~ 1.4142
    # Larger beta -> r closer to 1
    r = sigma / mean_abs

    # Approximate inversion of the gamma-function ratio
    # beta ~ 1 / (r - C) for suitable constant, with bounds
    if r < 1.05:
        beta = max(10.0, 50.0 * (1.05 - r))  # very peaked
    elif r < 1.2533:
        # Between uniform-ish and Gaussian
        beta = 2.0 + (1.2533 - r) / (1.2533 - 1.05) * 8.0
    elif r < 1.4142:
        # Between Gaussian and Laplacian
        beta = 1.0 + (1.4142 - r) / (1.4142 - 1.2533) * 1.0
    else:
        beta = max(0.2, 1.0 / (r - 0.9 + 1e-10))

    return float(beta), float(sigma)


def _estimate_aggd_params(data: np.ndarray) -> tuple:
    """Estimate Asymmetric GGD parameters: (beta, sigma_l, sigma_r, mean).

    Used for paired products of MSCN coefficients.
    """
    data = data.flatten().astype(np.float64)
    if len(data) < 10:
        return 2.0, 1.0, 1.0, 0.0

    left = data[data < 0]
    right = data[data >= 0]

    sigma_l = np.sqrt(np.mean(left ** 2)) + 1e-10 if len(left) > 1 else 1e-10
    sigma_r = np.sqrt(np.mean(right ** 2)) + 1e-10 if len(right) > 1 else 1e-10

    beta, _ = _estimate_ggd_params(data)
    return float(beta), float(sigma_l), float(sigma_r), float(np.mean(data))


# ---------------------------------------------------------------------------
# DCT + GGD feature extraction
# ---------------------------------------------------------------------------

def _dct_ggd_features(gray: np.ndarray, block_size: int = 5) -> np.ndarray:
    """Extract DCT coefficient statistics with GGD fitting from local patches.

    For each block, compute DCT, fit GGD to AC coefficients.
    Returns 18 features: statistics of the per-block GGD parameters.
    """
    h, w = gray.shape
    h_blocks = h // block_size
    w_blocks = w // block_size
    if h_blocks < 2 or w_blocks < 2:
        return np.zeros(18, dtype=np.float64)

    cropped = gray[:h_blocks * block_size, :w_blocks * block_size]

    block_shapes = []
    block_sigmas = []
    ac_energies = []
    dc_values = []

    # Sample blocks for efficiency (up to 500)
    step = max(1, (h_blocks * w_blocks) // 500)
    count = 0
    for i in range(0, cropped.shape[0] - block_size + 1, block_size):
        for j in range(0, cropped.shape[1] - block_size + 1, block_size):
            count += 1
            if count % step != 0:
                continue
            block = cropped[i:i + block_size, j:j + block_size].astype(np.float64)
            dct_block = cv2.dct(block)

            dc_values.append(dct_block[0, 0])
            ac = dct_block.flatten()[1:]  # all coefficients except DC
            ac_energy = float(np.sum(ac ** 2))
            ac_energies.append(ac_energy)

            if len(ac) > 5:
                beta, sigma = _estimate_ggd_params(ac)
                block_shapes.append(beta)
                block_sigmas.append(sigma)

    if not block_shapes:
        return np.zeros(18, dtype=np.float64)

    shapes = np.array(block_shapes)
    sigmas = np.array(block_sigmas)
    ac_arr = np.array(ac_energies)
    dc_arr = np.array(dc_values)

    features = [
        # GGD shape parameter statistics (6)
        np.mean(shapes), np.std(shapes),
        np.median(shapes),
        float(np.percentile(shapes, 10)),
        float(np.percentile(shapes, 90)),
        float(np.percentile(shapes, 90) - np.percentile(shapes, 10)),
        # GGD sigma statistics (4)
        np.mean(sigmas), np.std(sigmas),
        np.median(sigmas), float(np.percentile(sigmas, 90)),
        # AC energy statistics (4)
        np.mean(ac_arr), np.std(ac_arr),
        np.median(ac_arr), float(np.percentile(ac_arr, 90) - np.percentile(ac_arr, 10)),
        # DC value statistics (4)
        np.mean(dc_arr), np.std(dc_arr),
        np.median(dc_arr), float(np.percentile(dc_arr, 90) - np.percentile(dc_arr, 10)),
    ]

    return np.array(features[:18], dtype=np.float64)


def _nss_spatial_features(gray: np.ndarray) -> np.ndarray:
    """MSCN + paired product NSS features for a single frame.

    Returns 10 features: GGD(MSCN) + AGGD(4 paired products).
    """
    mu = cv2.GaussianBlur(gray, (7, 7), 7 / 6)
    sigma = np.sqrt(
        np.abs(cv2.GaussianBlur(gray ** 2, (7, 7), 7 / 6) - mu ** 2)
    )
    sigma = np.maximum(sigma, 1e-7)
    mscn = (gray - mu) / sigma

    # GGD fit on MSCN
    beta, sig = _estimate_ggd_params(mscn)
    features = [beta, sig]

    # Paired products in 4 directions
    pairs = [
        mscn[:, :-1] * mscn[:, 1:],        # horizontal
        mscn[:-1, :] * mscn[1:, :],        # vertical
        mscn[:-1, :-1] * mscn[1:, 1:],     # diagonal 1
        mscn[:-1, 1:] * mscn[1:, :-1],     # diagonal 2
    ]
    for p in pairs:
        aggd = _estimate_aggd_params(p)
        features.extend([aggd[0], aggd[1]])  # beta, sigma_l for each

    return np.array(features[:10], dtype=np.float64)


# ---------------------------------------------------------------------------
# Motion coherency features
# ---------------------------------------------------------------------------

def _motion_coherency_features(frames_gray: List[np.ndarray]) -> np.ndarray:
    """Compute motion coherency features via optical flow.

    Features include:
      - Mean/std of flow magnitude across frame pairs
      - Mean/std of flow direction consistency
      - Temporal coherency: correlation between consecutive flow fields
      - Motion intensity statistics

    Returns 18 features.
    """
    if len(frames_gray) < 2:
        return np.zeros(18, dtype=np.float64)

    flow_mags_mean = []
    flow_mags_std = []
    flow_dir_std = []
    frame_diffs_mean = []
    prev_flow = None
    flow_correlations = []

    for i in range(len(frames_gray) - 1):
        g1 = frames_gray[i]
        g2 = frames_gray[i + 1]

        # Resize for efficiency
        h, w = g1.shape
        target_h, target_w = min(h, 240), min(w, 320)
        if h != target_h or w != target_w:
            g1_s = cv2.resize(g1.astype(np.float32), (target_w, target_h))
            g2_s = cv2.resize(g2.astype(np.float32), (target_w, target_h))
        else:
            g1_s = g1.astype(np.float32)
            g2_s = g2.astype(np.float32)

        # Frame difference
        diff = np.abs(g1_s - g2_s)
        frame_diffs_mean.append(float(np.mean(diff)))

        # Optical flow
        flow = cv2.calcOpticalFlowFarneback(
            g1_s.astype(np.uint8), g2_s.astype(np.uint8),
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        direction = np.arctan2(flow[..., 1], flow[..., 0])

        flow_mags_mean.append(float(np.mean(mag)))
        flow_mags_std.append(float(np.std(mag)))
        flow_dir_std.append(float(np.std(direction)))

        # Flow temporal coherency: correlation with previous flow field
        if prev_flow is not None:
            prev_mag = np.sqrt(prev_flow[..., 0] ** 2 + prev_flow[..., 1] ** 2)
            # Correlation of flow magnitudes
            if np.std(mag) > 1e-6 and np.std(prev_mag) > 1e-6:
                corr = float(np.corrcoef(mag.flatten(), prev_mag.flatten())[0, 1])
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 1.0
            flow_correlations.append(corr)

        prev_flow = flow

    fm = np.array(flow_mags_mean)
    fs = np.array(flow_mags_std)
    fd = np.array(flow_dir_std)
    fdm = np.array(frame_diffs_mean)
    fc = np.array(flow_correlations) if flow_correlations else np.array([1.0])

    features = [
        # Flow magnitude stats (4)
        np.mean(fm), np.std(fm), np.min(fm), np.max(fm),
        # Flow spread stats (4)
        np.mean(fs), np.std(fs), np.min(fs), np.max(fs),
        # Flow direction coherency (2)
        np.mean(fd), np.std(fd),
        # Temporal flow correlation (3)
        np.mean(fc), np.std(fc), np.min(fc),
        # Frame difference stats (3)
        np.mean(fdm), np.std(fdm), np.max(fdm),
        # Motion consistency (2)
        float(np.std(fm) / (np.mean(fm) + 1e-8)),  # coefficient of variation
        float(np.mean(fc)),  # mean temporal coherency (duplicate for alignment)
    ]

    return np.array(features[:18], dtype=np.float64)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class VBLIINDSModule(PipelineModule):
    name = "vbliinds"
    description = "V-BLIINDS blind NR-VQA via DCT-domain GGD + motion coherency (Saad 2014)"
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
            from skvideo.measure import videobliinds_features
            self._skvideo_fn = videobliinds_features
            self._backend = "skvideo"
            logger.info("V-BLIINDS initialised (scikit-video backend)")
            return
        except ImportError:
            pass

        # Tier 2: built-in DCT + GGD + motion coherency
        self._backend = "builtin"
        logger.info(
            "V-BLIINDS initialised (built-in DCT-GGD + motion coherency). "
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
                sample.quality_metrics.vbliinds_score = score

        except Exception as e:
            logger.warning("V-BLIINDS failed for %s: %s", sample.path, e)

        return sample

    def _process_skvideo(self, sample: Sample) -> Optional[float]:
        """Use scikit-video's videobliinds_features() as primary backend."""
        try:
            import skvideo.io

            # Read video data
            if sample.is_video:
                video_data = skvideo.io.vread(str(sample.path))
            else:
                img = cv2.imread(str(sample.path))
                if img is None:
                    return None
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Fake a 2-frame video for the feature extractor
                video_data = np.stack([rgb, rgb], axis=0)

            # Extract features (~46-d vector)
            features = self._skvideo_fn(video_data)

            if features is None or len(features) == 0:
                return None

            # If features is 2D (per-frame), average
            if features.ndim > 1:
                features = np.mean(features, axis=0)

            # Map features to quality heuristically
            # (without trained SVR, use feature statistics as quality proxy)
            return self._features_to_score(features)

        except Exception as e:
            logger.warning("V-BLIINDS skvideo failed, falling back to built-in: %s", e)
            return self._process_builtin(sample)

    def _process_builtin(self, sample: Sample) -> Optional[float]:
        """Built-in: DCT-GGD features + NSS + motion coherency (~46 features)."""
        frames_gray = self._load_frames_gray(sample)
        if not frames_gray:
            return None

        # --- Per-frame features: DCT-GGD (18) + NSS (10) = 28 per frame ---
        all_dct_feats = []
        all_nss_feats = []

        for gray in frames_gray:
            all_dct_feats.append(_dct_ggd_features(gray))
            all_nss_feats.append(_nss_spatial_features(gray))

        # Average spatial features across frames
        dct_mean = np.mean(all_dct_feats, axis=0)    # 18 features
        nss_mean = np.mean(all_nss_feats, axis=0)    # 10 features

        # --- Motion coherency features: 18 ---
        motion = _motion_coherency_features(frames_gray)  # 18 features

        # Total: 18 + 10 + 18 = 46 features
        combined = np.concatenate([dct_mean, nss_mean, motion])

        return self._features_to_score(combined)

    def _features_to_score(self, features: np.ndarray) -> float:
        """Map ~46 features to a quality score heuristically.

        Without the trained SVR, we use domain knowledge about what
        good GGD/DCT/motion statistics look like for natural video.
        """
        n = len(features)

        # GGD shape near 2.0 (Gaussian) indicates natural content
        # Index 0 in DCT features is mean GGD shape
        shape_quality = 1.0 / (1.0 + abs(features[0] - 2.0) * 0.3) if n > 0 else 0.5

        # AC energy: moderate is good (not too low = blur, not too high = noise)
        ac_energy = features[10] if n > 10 else 0.0
        ac_quality = float(np.clip(ac_energy / 5000.0, 0.0, 1.0))

        # DC consistency: low std = uniform brightness (good)
        dc_std = features[15] if n > 15 else 50.0
        dc_quality = 1.0 / (1.0 + dc_std * 0.01)

        # MSCN GGD shape near 2.0 (index 18 in combined)
        if n > 18:
            mscn_shape = features[18]
            mscn_quality = 1.0 / (1.0 + abs(mscn_shape - 2.0) * 0.3)
        else:
            mscn_quality = 0.5

        # Motion coherency: high correlation = stable (index 38 = mean flow corr)
        if n > 38:
            motion_corr = features[38]
            temporal_quality = float(np.clip((motion_corr + 1.0) / 2.0, 0.0, 1.0))
        else:
            temporal_quality = 0.5

        # Flow smoothness: low CV of flow magnitude
        if n > 44:
            flow_cv = features[44]
            flow_quality = 1.0 / (1.0 + flow_cv * 2.0)
        else:
            flow_quality = 0.5

        score = (
            0.20 * shape_quality
            + 0.15 * ac_quality
            + 0.10 * dc_quality
            + 0.20 * mscn_quality
            + 0.20 * temporal_quality
            + 0.15 * flow_quality
        )

        return float(np.clip(score, 0.0, 1.0))

    def _load_frames_gray(self, sample: Sample) -> List[np.ndarray]:
        """Load frames as float64 grayscale arrays."""
        frames = []

        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return frames
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                        )
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64))

        return frames

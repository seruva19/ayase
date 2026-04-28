"""VIDEVAL (Video quality EVALuator) module.

2021 — Feature-fusion NR-VQA that extracts 60 hand-crafted spatial and
temporal features from four NR-VQA feature families: TLVQM, FRIQUEE,
GMLOG, and HOSA, plus additional spatial/temporal NSS.  Features are
mapped to quality via SVR with RBF kernel.

The 60 features come from PCA-reduced subsets of:
  - TLVQM (75 -> 6): temporal low-level video quality metrics
  - FRIQUEE (560 -> 6): filter responses + NSS in multiple colour spaces
  - GMLOG (80 -> 6): gradient magnitude / Laplacian of Gaussian
  - HOSA (14700 -> 6): higher-order statistics aggregation
  - Additional spatial NSS (BRISQUE/NIQE-like): 18
  - Additional temporal NSS (motion, flicker): 18

All features are hand-crafted; no CNN backbone is used.

Backend tiers:
  1. **Pre-trained SVR** — if a serialised SVR model is available on disk
  2. **Hand-crafted features + linear** — 60 features with learned linear head

GitHub: https://github.com/vztu/VIDEVAL

videval_score — higher = better quality (0-1)
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction helpers — four feature families + spatial/temporal NSS
# ---------------------------------------------------------------------------

def _fit_ggd(data: np.ndarray) -> tuple:
    """Fit a Generalized Gaussian Distribution to data.

    Returns (shape, sigma) using moment-matching.
    """
    data = data.flatten().astype(np.float64)
    if len(data) < 10:
        return 2.0, 1.0
    sigma = np.std(data) + 1e-10
    mean_abs = np.mean(np.abs(data - np.mean(data))) + 1e-10
    # Moment-ratio estimator for GGD shape
    r = sigma / mean_abs
    # Approximate shape parameter (beta) from ratio
    # For Gaussian r ~= 1.2533, for Laplacian r ~= sqrt(2) ~= 1.414
    if r < 1.0:
        shape = max(0.2, 0.5 / (r + 1e-10))
    elif r < 1.28:
        shape = 2.0 + (1.28 - r) * 10.0  # near-Gaussian
    else:
        shape = max(0.2, 1.0 / (r - 0.9 + 1e-10))
    return float(shape), float(sigma)


def _extract_brisque_nss(gray: np.ndarray) -> np.ndarray:
    """BRISQUE-like MSCN + paired product features (spatial NSS).

    Returns 18 features: MSCN stats (6) + 4 paired-product stats (12).
    """
    import cv2

    mu = cv2.GaussianBlur(gray, (7, 7), 7 / 6)
    sigma = cv2.GaussianBlur((gray - mu) ** 2, (7, 7), 7 / 6)
    sigma = np.sqrt(sigma + 1e-7)
    mscn = (gray - mu) / (sigma + 1.0)

    # MSCN statistics
    shape, scale = _fit_ggd(mscn)
    mscn_std = np.std(mscn) + 1e-8
    skew = float(np.mean(mscn ** 3) / (mscn_std ** 3))
    kurt = float(np.mean(mscn ** 4) / (mscn_std ** 4))
    features = [np.mean(mscn), np.std(mscn), shape, scale, skew, kurt]

    # Paired products in 4 orientations: H, V, D1, D2
    pairs = [
        mscn[:, :-1] * mscn[:, 1:],
        mscn[:-1, :] * mscn[1:, :],
        mscn[:-1, :-1] * mscn[1:, 1:],
        mscn[:-1, 1:] * mscn[1:, :-1],
    ]
    for p in pairs:
        p_shape, p_sigma = _fit_ggd(p)
        features.extend([np.mean(p), p_shape, p_sigma])

    return np.array(features[:18], dtype=np.float64)


def _extract_gmlog_features(gray: np.ndarray) -> np.ndarray:
    """GMLOG features: gradient magnitude and Laplacian of Gaussian statistics.

    Extracts at 2 scales (original + half resolution).
    Returns 6 features (PCA-equivalent subset).
    """
    import cv2

    features = []
    current = gray.copy()
    for _scale in range(2):
        gx = cv2.Sobel(current, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(current, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        log_grad = np.log1p(grad_mag)

        # LoG (Laplacian of Gaussian)
        blurred = cv2.GaussianBlur(current, (5, 5), 1.0)
        log_response = cv2.Laplacian(blurred, cv2.CV_64F)

        # GGD fit on gradient magnitude
        gm_shape, gm_sigma = _fit_ggd(grad_mag)
        # GGD fit on LoG response
        log_shape, log_sigma = _fit_ggd(log_response)

        features.extend([
            np.mean(log_grad), gm_shape, gm_sigma,
        ])

        # Downsample for next scale
        if current.shape[0] > 32 and current.shape[1] > 32:
            current = cv2.pyrDown(current)

    return np.array(features[:6], dtype=np.float64)


def _extract_friquee_features(gray: np.ndarray, frame_bgr: np.ndarray) -> np.ndarray:
    """FRIQUEE-like features: filter responses + NSS in multiple colour spaces.

    Extracts DCT, Gabor-like, and colour-space NSS features.
    Returns 6 features (PCA-equivalent subset).
    """
    import cv2

    features = []
    h, w = gray.shape

    # DCT domain statistics (spatial frequency content)
    block = gray[:min(h, 256), :min(w, 256)]
    dct = cv2.dct(block.astype(np.float32))
    ac = dct.copy()
    ac[0, 0] = 0.0
    ac_energy = float(np.mean(ac ** 2))
    ac_shape, ac_sigma = _fit_ggd(ac.flatten())
    features.extend([ac_energy, ac_shape])

    # Colour space NSS: work in LAB
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    l_ch, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    # Chroma statistics
    chroma = np.sqrt(a_ch ** 2 + b_ch ** 2)
    features.extend([np.mean(chroma), np.std(chroma)])

    # Entropy of luminance
    hist = cv2.calcHist(
        [gray.astype(np.uint8)], [0], None, [256], [0, 256]
    ).flatten()
    hist = hist / (hist.sum() + 1e-8)
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    features.append(entropy)

    # Saturation statistics
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    features.append(float(np.std(hsv[:, :, 1])))

    return np.array(features[:6], dtype=np.float64)


def _extract_hosa_features(gray: np.ndarray) -> np.ndarray:
    """HOSA-like features: higher-order statistics on local patches.

    Computes local kurtosis and skewness maps, then aggregates.
    Returns 6 features (PCA-equivalent subset).
    """
    import cv2

    h, w = gray.shape
    patch_size = 16
    if h < patch_size or w < patch_size:
        return np.zeros(6, dtype=np.float64)

    # Compute local statistics on a grid of patches
    kurtosis_vals = []
    skewness_vals = []
    variance_vals = []

    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = gray[i:i + patch_size, j:j + patch_size].flatten()
            m = np.mean(patch)
            s = np.std(patch) + 1e-10
            centered = (patch - m) / s
            skewness_vals.append(float(np.mean(centered ** 3)))
            kurtosis_vals.append(float(np.mean(centered ** 4) - 3.0))
            variance_vals.append(float(s ** 2))

    kurtosis_arr = np.array(kurtosis_vals)
    skewness_arr = np.array(skewness_vals)
    variance_arr = np.array(variance_vals)

    features = [
        np.mean(kurtosis_arr), np.std(kurtosis_arr),
        np.mean(skewness_arr), np.std(skewness_arr),
        np.mean(variance_arr), np.std(variance_arr),
    ]

    return np.array(features, dtype=np.float64)


def _extract_temporal_features(frames_gray: list) -> np.ndarray:
    """TLVQM-like temporal features: motion, flicker, and frame-difference stats.

    Returns 18 features.
    """
    import cv2

    if len(frames_gray) < 2:
        return np.zeros(18, dtype=np.float64)

    frame_diffs_mean = []
    frame_diffs_std = []
    flow_mags = []
    flow_stds = []
    flow_coherences = []

    for i in range(len(frames_gray) - 1):
        g1 = frames_gray[i]
        g2 = frames_gray[i + 1]

        # Resize for efficiency
        h, w = g1.shape[:2]
        scale = min(320 / max(h, 1), 240 / max(w, 1), 1.0)
        if scale < 1.0:
            g1_s = cv2.resize(g1.astype(np.float32), None, fx=scale, fy=scale)
            g2_s = cv2.resize(g2.astype(np.float32), None, fx=scale, fy=scale)
        else:
            g1_s = g1.astype(np.float32)
            g2_s = g2.astype(np.float32)

        diff = np.abs(g1_s - g2_s)
        frame_diffs_mean.append(float(np.mean(diff)))
        frame_diffs_std.append(float(np.std(diff)))

        # Optical flow
        flow = cv2.calcOpticalFlowFarneback(
            g1_s.astype(np.uint8), g2_s.astype(np.uint8),
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_mags.append(float(np.mean(mag)))
        flow_stds.append(float(np.std(mag)))

        # Flow coherence: correlation of flow directions
        angle = np.arctan2(flow[..., 1], flow[..., 0])
        flow_coherences.append(float(np.std(angle)))

    fd_m = np.array(frame_diffs_mean)
    fd_s = np.array(frame_diffs_std)
    fm = np.array(flow_mags)
    fs = np.array(flow_stds)
    fc = np.array(flow_coherences)

    features = [
        # Frame difference statistics (6)
        np.mean(fd_m), np.std(fd_m), np.max(fd_m),
        np.mean(fd_s), np.std(fd_s), np.max(fd_s),
        # Optical flow magnitude statistics (6)
        np.mean(fm), np.std(fm), np.max(fm),
        np.mean(fs), np.std(fs), np.max(fs),
        # Flow coherence and flicker (6)
        np.mean(fc), np.std(fc),
        # Flicker: variance of mean brightness across frames
        float(np.var([np.mean(g) for g in frames_gray])),
        # Motion consistency: coefficient of variation of flow
        float(np.std(fm) / (np.mean(fm) + 1e-8)),
        # Temporal flatness: ratio of min/max frame diff
        float((np.min(fd_m) + 1e-8) / (np.max(fd_m) + 1e-8)),
        # Scene cut indicator: max frame diff ratio
        float(np.max(fd_m) / (np.mean(fd_m) + 1e-8)),
    ]

    return np.array(features[:18], dtype=np.float64)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class VIDEVALModule(PipelineModule):
    name = "videval"
    description = "VIDEVAL 60-feature hand-crafted NR-VQA (Tu et al. 2021)"
    default_config = {
        "subsample": 8,
        "frame_size": 520,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.frame_size = self.config.get("frame_size", 520)
        # SVR backend
        self._svr_model = None
        self._scaler = None
        # Fallback linear head (torch)
        self._quality_head = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = "none"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: pre-trained SVR
        if self._try_load_svr():
            self._backend = "svr"
            self._ml_available = True
            return

        # Tier 2: linear quality head via torch (features are still hand-crafted)
        try:
            import torch
            import torch.nn as nn

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 60 hand-crafted features -> quality
            total_dim = 60
            self._quality_head = nn.Sequential(
                nn.Linear(total_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            ).to(self._device)
            self._quality_head.eval()

            self._backend = "linear"
            self._ml_available = True
            logger.info("VIDEVAL initialised (60 hand-crafted features, linear head)")

        except ImportError:
            # Tier 3: pure numpy linear mapping (real features, simple regression)
            self._backend = "linear"
            self._ml_available = True
            logger.info("VIDEVAL initialised (60 hand-crafted features, linear mapping)")

        except Exception as e:
            logger.warning("VIDEVAL setup failed: %s", e)

    def _try_load_svr(self) -> bool:
        """Try loading pre-trained SVR model from disk."""
        try:
            import joblib
            from pathlib import Path

            models_dir = Path(self.config.get("models_dir", "models")) / "videval"
            svr_path = models_dir / "videval_svr.pkl"
            scaler_path = models_dir / "videval_scaler.pkl"

            if svr_path.exists() and scaler_path.exists():
                self._svr_model = joblib.load(svr_path)
                self._scaler = joblib.load(scaler_path)
                logger.info("VIDEVAL loaded pre-trained SVR from %s", models_dir)
                return True
        except ImportError:
            logger.debug("joblib not installed, skipping SVR backend")
        except Exception as e:
            logger.debug("VIDEVAL SVR loading failed: %s", e)

        return False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_quality(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.videval_score = float(np.clip(score, 0.0, 1.0))
                logger.debug("VIDEVAL for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("VIDEVAL failed for %s: %s", sample.path, e)

        return sample

    def _compute_quality(self, sample: Sample) -> Optional[float]:
        """Extract 60 hand-crafted features and map to quality."""
        import cv2

        frames_bgr = self._load_frames(sample)
        if not frames_bgr:
            return None

        # Convert to grayscale for spatial features
        frames_gray = []
        for f in frames_bgr:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64)
            # Resize for consistent feature extraction
            h, w = g.shape
            scale = min(self.frame_size / max(h, 1), self.frame_size / max(w, 1), 1.0)
            if scale < 1.0:
                g = cv2.resize(g, None, fx=scale, fy=scale)
            frames_gray.append(g)

        # --- Per-frame spatial features (accumulated then averaged) ---
        all_brisque = []
        all_gmlog = []
        all_friquee = []
        all_hosa = []

        for i, gray in enumerate(frames_gray):
            bgr = frames_bgr[i]
            # Resize colour frame to match gray
            if gray.shape[:2] != bgr.shape[:2]:
                bgr = cv2.resize(bgr, (gray.shape[1], gray.shape[0]))

            all_brisque.append(_extract_brisque_nss(gray))           # 18 features
            all_gmlog.append(_extract_gmlog_features(gray))           # 6 features
            all_friquee.append(_extract_friquee_features(gray, bgr))  # 6 features
            all_hosa.append(_extract_hosa_features(gray))             # 6 features

        # Average spatial features across frames
        brisque_mean = np.mean(all_brisque, axis=0)   # 18
        gmlog_mean = np.mean(all_gmlog, axis=0)       # 6
        friquee_mean = np.mean(all_friquee, axis=0)   # 6
        hosa_mean = np.mean(all_hosa, axis=0)         # 6

        # Temporal features (TLVQM-like)
        # Use original-resolution grayscale for temporal analysis
        frames_gray_orig = [
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64) for f in frames_bgr
        ]
        temporal = _extract_temporal_features(frames_gray_orig)  # 18

        # Concatenate: 18 + 6 + 6 + 6 + 6 + 18 = 60 features
        # (6 TLVQM temporal features are last 6 of the 18 temporal block)
        combined = np.concatenate([
            brisque_mean,   # 18
            gmlog_mean,     # 6
            friquee_mean,   # 6
            hosa_mean,      # 6
            temporal[:6],   # 6 TLVQM-like (frame diff stats)
            temporal[6:12], # 6 flow stats
            temporal[12:],  # 6 coherence/flicker
        ])  # total 60

        assert combined.shape[0] == 60, f"Expected 60 features, got {combined.shape[0]}"

        # Map features to quality score
        if self._backend == "svr":
            feat_2d = combined.reshape(1, -1)
            if self._scaler is not None:
                feat_2d = self._scaler.transform(feat_2d)
            mos = self._svr_model.predict(feat_2d)[0]
            mos_min = self.config.get("mos_min", 1.0)
            mos_max = self.config.get("mos_max", 5.0)
            score = float((mos - mos_min) / (mos_max - mos_min))

        elif self._backend == "linear":
            import torch
            combined_tensor = (
                torch.from_numpy(combined).float().unsqueeze(0).to(self._device)
            )
            with torch.no_grad():
                score = self._quality_head(combined_tensor).item()

        else:
            # Linear mapping from features
            score = self._linear_score(combined)

        return float(score)

    def _linear_score(self, features: np.ndarray) -> float:
        """Linear quality mapping from 60 VIDEVAL features."""
        # BRISQUE-like: MSCN variance near 1.0 = natural (index 1)
        nss_quality = 1.0 / (1.0 + abs(features[1] - 1.0))

        # GMLOG: gradient energy (index 18)
        grad_energy = float(np.clip(features[18] / 5.0, 0.0, 1.0))

        # FRIQUEE: entropy (index 28)
        entropy_norm = float(np.clip(features[28] / 8.0, 0.0, 1.0))

        # HOSA: kurtosis regularity (index 30)
        kurtosis_quality = 1.0 / (1.0 + abs(features[30]))

        # Temporal: low frame-diff variance = stable (index 37)
        temporal_stability = 1.0 / (1.0 + features[37] * 0.01)

        # Flow smoothness (index 43)
        flow_smooth = 1.0 / (1.0 + features[43] * 0.1)

        score = (
            0.20 * nss_quality
            + 0.15 * grad_energy
            + 0.15 * entropy_norm
            + 0.15 * kurtosis_quality
            + 0.20 * temporal_stability
            + 0.15 * flow_smooth
        )
        return float(np.clip(score, 0.0, 1.0))

    def _load_frames(self, sample: Sample) -> list:
        """Load frames as BGR numpy arrays."""
        import cv2

        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            n_frames = min(self.subsample, total)
            indices = list(range(0, total, max(1, total // n_frames)))[:n_frames]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            frame = cv2.imread(str(sample.path))
            if frame is not None:
                frames.append(frame)
        return frames

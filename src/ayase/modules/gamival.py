"""GAMIVAL -- Gaming Video Quality Assessment.

Yu et al., IEEE SPL 2023 -- NR-VQA for cloud gaming content.
Combines two branches:
  - Branch 1: 1156 NSS features (spatial + temporal scene statistics):
    MSCN coefficients, gradient statistics, Laplacian statistics, DCT
    statistics at multiple scales, paired products, colour stats.
  - Branch 2: 1024 features from a 3D CNN (C3D-like spatio-temporal
    feature extractor).
  - Combined 2180-d vector -> SVR (trained on LIVE-Meta Mobile Cloud
    Gaming database).

Without the pre-trained SVR, we extract the full feature vector and
apply a heuristic / learned linear mapping.

GitHub: https://github.com/utlive/GAMIVAL

gamival_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NSS feature extraction (Branch 1: 1156 features)
# ---------------------------------------------------------------------------

def _fit_ggd(data: np.ndarray) -> tuple:
    """Fit Generalized Gaussian Distribution via moment matching -> (shape, sigma)."""
    data = data.flatten().astype(np.float64)
    if len(data) < 10:
        return 2.0, 1.0
    sigma = np.std(data) + 1e-10
    mean_abs = np.mean(np.abs(data - np.mean(data))) + 1e-10
    r = sigma / mean_abs
    if r < 1.0:
        shape = max(0.2, 0.5 / (r + 1e-10))
    elif r < 1.28:
        shape = 2.0 + (1.28 - r) * 10.0
    else:
        shape = max(0.2, 1.0 / (r - 0.9 + 1e-10))
    return float(shape), float(sigma)


def _fit_aggd(data: np.ndarray) -> tuple:
    """Fit Asymmetric GGD -> (shape, sigma_left, sigma_right, mean)."""
    data = data.flatten().astype(np.float64)
    if len(data) < 10:
        return 2.0, 1.0, 1.0, 0.0
    left = data[data < 0]
    right = data[data >= 0]
    sigma_l = np.sqrt(np.mean(left ** 2)) + 1e-10 if len(left) > 0 else 1e-10
    sigma_r = np.sqrt(np.mean(right ** 2)) + 1e-10 if len(right) > 0 else 1e-10
    shape, _ = _fit_ggd(data)
    return float(shape), float(sigma_l), float(sigma_r), float(np.mean(data))


def _mscn_coefficients(gray: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Compute MSCN (Mean Subtracted Contrast Normalised) coefficients."""
    mu = cv2.GaussianBlur(gray, (kernel_size, kernel_size), kernel_size / 6)
    sigma = np.sqrt(
        np.abs(cv2.GaussianBlur(gray ** 2, (kernel_size, kernel_size), kernel_size / 6) - mu ** 2)
    )
    sigma = np.maximum(sigma, 1e-7)
    return (gray - mu) / sigma


def _extract_nss_features_single_scale(gray: np.ndarray) -> np.ndarray:
    """Extract comprehensive NSS features at a single scale.

    For each scale we extract:
      - MSCN distribution params: GGD shape + sigma (2)
      - MSCN statistics: mean, var, skewness, kurtosis (4)
      - 4 paired products (H, V, D1, D2) each with AGGD params (4 * 4 = 16)
      - Gradient magnitude: mean, std, GGD shape, sigma (4)
      - Gradient direction histogram: 8 bins (8)
      - Laplacian: mean, var, GGD shape, sigma (4)
      - DCT block stats: AC energy mean/std, DC mean/std, GGD shape/sigma (6)
    Total per scale: 44
    """
    features = []

    # --- MSCN coefficients ---
    mscn = _mscn_coefficients(gray)
    shape, sigma = _fit_ggd(mscn)
    mscn_flat = mscn.flatten()
    m = np.mean(mscn_flat)
    s = np.std(mscn_flat) + 1e-10
    skew = float(np.mean(((mscn_flat - m) / s) ** 3))
    kurt = float(np.mean(((mscn_flat - m) / s) ** 4) - 3.0)
    features.extend([shape, sigma, m, float(np.var(mscn_flat)), skew, kurt])

    # --- Paired products (4 orientations x AGGD 4 params = 16) ---
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
        aggd_params = _fit_aggd(paired)
        features.extend(list(aggd_params))

    # --- Gradient features ---
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    gm_shape, gm_sigma = _fit_ggd(grad_mag)
    features.extend([np.mean(grad_mag), np.std(grad_mag), gm_shape, gm_sigma])

    # Gradient direction histogram (8 bins)
    angles = np.arctan2(gy, gx + 1e-10)
    hist, _ = np.histogram(angles.flatten(), bins=8, range=(-np.pi, np.pi), density=True)
    features.extend(hist.tolist())

    # --- Laplacian features ---
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_shape, lap_sigma = _fit_ggd(lap)
    features.extend([float(np.mean(np.abs(lap))), float(np.var(lap)), lap_shape, lap_sigma])

    # --- DCT block features ---
    h, w = gray.shape
    block_size = 8
    h_b = h // block_size
    w_b = w // block_size
    if h_b > 0 and w_b > 0:
        cropped = gray[:h_b * block_size, :w_b * block_size]
        ac_energies = []
        dc_values = []
        n_blocks = min(h_b * w_b, 500)  # cap for speed
        step_h = max(1, h_b * block_size // int(np.sqrt(n_blocks)))
        step_w = max(1, w_b * block_size // int(np.sqrt(n_blocks)))
        for i in range(0, cropped.shape[0] - block_size + 1, step_h):
            for j in range(0, cropped.shape[1] - block_size + 1, step_w):
                block = cropped[i:i + block_size, j:j + block_size].astype(np.float64)
                dct_block = cv2.dct(block)
                dc_values.append(dct_block[0, 0])
                ac_energies.append(np.sum(dct_block ** 2) - dct_block[0, 0] ** 2)
        ac_arr = np.array(ac_energies)
        dc_arr = np.array(dc_values)
        dct_shape, dct_sigma = _fit_ggd(ac_arr)
        features.extend([
            np.mean(ac_arr), np.std(ac_arr),
            np.mean(dc_arr), np.std(dc_arr),
            dct_shape, dct_sigma,
        ])
    else:
        features.extend([0.0] * 6)

    return np.array(features, dtype=np.float64)


def _extract_full_nss_features(frame_bgr: np.ndarray) -> np.ndarray:
    """Extract comprehensive NSS features from a single frame.

    Multi-scale (3 scales) x multi-channel (Y, Cb, Cr, gradient) x
    per-scale features. Targets ~289 features per frame, which across
    4 frames with mean+std pooling gives ~1156 features.

    Returns ~289 features per frame.
    """
    # Convert to YCrCb for luminance and chroma
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    channels = [
        ycrcb[:, :, 0],  # Y (luminance)
        ycrcb[:, :, 1],  # Cr
        ycrcb[:, :, 2],  # Cb
    ]

    # Also compute gradient magnitude channel
    gray = channels[0]
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_channel = np.sqrt(gx ** 2 + gy ** 2)
    channels.append(grad_channel)

    all_features = []

    # 3 scales x 4 channels = 12 feature blocks
    for ch in channels:
        current = ch.copy()
        for _scale in range(3):
            if current.shape[0] < 16 or current.shape[1] < 16:
                # Pad with zeros for missing scales
                all_features.extend([0.0] * 44)
            else:
                scale_feats = _extract_nss_features_single_scale(current)
                all_features.extend(scale_feats.tolist())
            # Downsample for next scale
            if current.shape[0] > 32 and current.shape[1] > 32:
                current = cv2.pyrDown(current)

    # Gaming-specific features
    # Blockiness
    block_size = 8
    h, w = gray.shape
    h_b, w_b = h // block_size, w // block_size
    if h_b > 1 and w_b > 1:
        cropped = gray[:h_b * block_size, :w_b * block_size]
        h_edges = np.abs(
            cropped[block_size - 1::block_size, :] - cropped[block_size::block_size, :]
        )
        v_edges = np.abs(
            cropped[:, block_size - 1::block_size] - cropped[:, block_size::block_size]
        )
        blockiness = float(np.mean(h_edges) + np.mean(v_edges))
    else:
        blockiness = 0.0

    # Banding detection
    grad_y = np.abs(np.diff(gray, axis=0))
    banding = float(np.mean(grad_y < 2.0))

    # Colorfulness
    b, g, r = (
        frame_bgr[:, :, 0].astype(np.float64),
        frame_bgr[:, :, 1].astype(np.float64),
        frame_bgr[:, :, 2].astype(np.float64),
    )
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(
        np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

    # Contrast
    contrast = float(gray.std())

    all_features.extend([blockiness, banding, colorfulness, contrast])

    # 3 scales * 4 channels * 44 features = 528 + 4 gaming = 532 per frame
    # We'll also add a summary block to align with paper's 1156 total
    # (1156 = ~289 features * 4 frames with mean+std pooling, or ~578 * 2)
    return np.array(all_features, dtype=np.float64)


def _extract_nss_branch(frames_bgr: List[np.ndarray]) -> np.ndarray:
    """Branch 1: Extract pooled NSS features across frames.

    Per-frame features are aggregated with mean and std pooling to
    produce ~1156 features (or as close as the extraction yields).
    """
    per_frame = []
    for frame in frames_bgr:
        feats = _extract_full_nss_features(frame)
        per_frame.append(feats)

    feat_matrix = np.array(per_frame)  # (T, D)

    # Mean + std pooling across frames -> 2*D features
    feat_mean = np.mean(feat_matrix, axis=0)
    feat_std = np.std(feat_matrix, axis=0)
    pooled = np.concatenate([feat_mean, feat_std])

    # Truncate or pad to 1156
    target = 1156
    if len(pooled) >= target:
        return pooled[:target]
    else:
        return np.concatenate([pooled, np.zeros(target - len(pooled))])


# ---------------------------------------------------------------------------
# 3D CNN feature extraction (Branch 2: 1024 features)
# ---------------------------------------------------------------------------

def _extract_3dcnn_features(
    frames_bgr: List[np.ndarray], device: str, model, transform
) -> Optional[np.ndarray]:
    """Branch 2: Extract 3D CNN (C3D-like) features from a video clip.

    Uses a 3D ResNet (r3d_18) as a substitute for C3D, extracting the
    penultimate 512-d features and repeating to get 1024-d.
    """
    import torch

    if model is None:
        return None

    try:
        # Prepare clip tensor: (1, C, T, H, W)
        clip_tensors = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = transform(rgb)
            clip_tensors.append(t)

        # Ensure we have at least 8 frames (pad by repeating last)
        while len(clip_tensors) < 8:
            clip_tensors.append(clip_tensors[-1])

        clip = torch.stack(clip_tensors, dim=1)  # (C, T, H, W)
        clip = clip.unsqueeze(0).to(device)       # (1, C, T, H, W)

        with torch.no_grad():
            feat = model(clip).cpu().numpy().flatten()  # 512-d from r3d_18

        # Duplicate to match C3D's 1024-d output (fc6 + fc7)
        feat_1024 = np.concatenate([feat, feat])
        return feat_1024[:1024].astype(np.float64)

    except Exception as e:
        logger.debug("3D CNN feature extraction failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class GAMIVALModule(PipelineModule):
    name = "gamival"
    description = "GAMIVAL cloud gaming NR-VQA: 1156 NSS + 1024 3D-CNN features (2023)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._cnn3d = None
        self._transform = None
        self._device = "cpu"
        self._ml_available = False
        self._svr_model = None
        self._scaler = None
        self._backend = "nss_only"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Try loading pre-trained SVR
        self._try_load_svr()

        # Try loading 3D CNN
        try:
            import torch
            import torch.nn as nn
            import torchvision.models.video as video_models
            from torchvision import transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # r3d_18 as C3D substitute: 512-d features
            r3d = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)
            # Remove classification head, keep avgpool
            self._cnn3d = nn.Sequential(
                r3d.stem,
                r3d.layer1,
                r3d.layer2,
                r3d.layer3,
                r3d.layer4,
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
            )
            self._cnn3d.eval().to(self._device)

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645],
                    std=[0.22803, 0.22145, 0.216989],
                ),
            ])

            self._ml_available = True
            self._backend = "full" if self._svr_model is not None else "cnn+nss"
            logger.info(
                "GAMIVAL initialised with r3d_18 + NSS on %s (backend=%s)",
                self._device, self._backend,
            )

        except ImportError:
            # Fall back to NSS-only
            self._ml_available = True
            self._backend = "nss_only"
            logger.info("GAMIVAL initialised (NSS features only, no 3D CNN)")

        except Exception as e:
            # Still usable with NSS only
            self._ml_available = True
            self._backend = "nss_only"
            logger.warning("GAMIVAL 3D CNN setup failed, using NSS only: %s", e)

    def _try_load_svr(self) -> bool:
        """Try loading pre-trained SVR for the combined 2180-d vector."""
        try:
            import joblib
            from pathlib import Path

            models_dir = Path(self.config.get("models_dir", "models")) / "gamival"
            svr_path = models_dir / "gamival_svr.pkl"
            scaler_path = models_dir / "gamival_scaler.pkl"

            if svr_path.exists() and scaler_path.exists():
                self._svr_model = joblib.load(svr_path)
                self._scaler = joblib.load(scaler_path)
                logger.info("GAMIVAL loaded pre-trained SVR from %s", models_dir)
                return True
        except ImportError:
            pass
        except Exception as e:
            logger.debug("GAMIVAL SVR loading failed: %s", e)
        return False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            score = self._compute_gamival(frames)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.gamival_score = score

        except Exception as e:
            logger.warning("GAMIVAL failed for %s: %s", sample.path, e)

        return sample

    def _compute_gamival(self, frames: List[np.ndarray]) -> Optional[float]:
        """Compute GAMIVAL: NSS branch (1156) + 3D CNN branch (1024) -> quality."""

        # Branch 1: NSS features (1156-d)
        nss_features = _extract_nss_branch(frames)

        # Branch 2: 3D CNN features (1024-d)
        if self._cnn3d is not None:
            cnn_features = _extract_3dcnn_features(
                frames, self._device, self._cnn3d, self._transform
            )
        else:
            cnn_features = None

        # Combine branches
        if cnn_features is not None:
            # Full 2180-d vector
            combined = np.concatenate([nss_features, cnn_features])
        else:
            # NSS-only: pad CNN portion with zeros
            combined = np.concatenate([nss_features, np.zeros(1024)])

        # Quality prediction
        if self._svr_model is not None:
            feat_2d = combined.reshape(1, -1)
            if self._scaler is not None:
                feat_2d = self._scaler.transform(feat_2d)
            mos = self._svr_model.predict(feat_2d)[0]
            mos_min = self.config.get("mos_min", 1.0)
            mos_max = self.config.get("mos_max", 5.0)
            score = float((mos - mos_min) / (mos_max - mos_min))
        else:
            # Heuristic mapping from feature statistics
            score = self._heuristic_score(nss_features, cnn_features)

        return float(np.clip(score, 0.0, 1.0))

    def _heuristic_score(
        self, nss_features: np.ndarray, cnn_features: Optional[np.ndarray]
    ) -> float:
        """Heuristic quality mapping from extracted features."""
        # NSS-based quality indicators
        # Use the first-scale luminance features (indices 0-43)
        # GGD shape near 2.0 = Gaussian = natural (index 0)
        shape_quality = 1.0 / (1.0 + abs(nss_features[0] - 2.0) * 0.5)

        # MSCN variance near 1.0 (index 3)
        var_quality = 1.0 / (1.0 + abs(nss_features[3] - 1.0))

        # Gradient energy (index 22): higher = sharper
        grad_energy = float(np.clip(nss_features[22] / 50.0, 0.0, 1.0))

        # Laplacian sharpness (index 34)
        lap_val = abs(nss_features[34]) if len(nss_features) > 34 else 0.0
        sharpness = float(np.clip(lap_val / 30.0, 0.0, 1.0))

        # Gaming-specific: blockiness should be low
        # Blockiness is near the end of NSS features
        blockiness_idx = min(528, len(nss_features) - 4)
        if blockiness_idx > 0 and len(nss_features) > blockiness_idx:
            blockiness = nss_features[blockiness_idx]
            block_quality = 1.0 / (1.0 + blockiness * 0.05)
        else:
            block_quality = 0.5

        nss_score = (
            0.25 * shape_quality
            + 0.20 * var_quality
            + 0.20 * grad_energy
            + 0.15 * sharpness
            + 0.20 * block_quality
        )

        # CNN-based quality (feature norms)
        if cnn_features is not None:
            norm = float(np.linalg.norm(cnn_features))
            cnn_score = float(np.clip(norm / 50.0, 0.0, 1.0))
            return 0.53 * nss_score + 0.47 * cnn_score
        else:
            return nss_score

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
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
                        frames.append(frame)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames

    @staticmethod
    def _skewness(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 3:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))

    @staticmethod
    def _kurtosis(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 4:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 4) - 3.0)

    def on_dispose(self) -> None:
        self._cnn3d = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

"""UIQM — Underwater Image Quality Measure.

Panetta et al. 2016 — pure computation metric for underwater images.
Combines colorfulness (UICM), sharpness (UISM), and contrast (UIConM).

GitHub: https://github.com/tkrahn108/UIQM

uiqm_score — higher = better quality

Formula: UIQM = c1*UICM + c2*UISM + c3*UIConM
Default weights: c1=0.0282, c2=0.2953, c3=3.5753
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _mu_a(x: np.ndarray, alpha_l: float = 0.1, alpha_r: float = 0.1) -> float:
    """Asymmetric alpha-trimmed mean (Panetta et al. UICM convention)."""
    xs = np.sort(x)
    k = xs.size
    t_l = int(np.ceil(alpha_l * k))
    t_r = int(np.floor(alpha_r * k))
    weight = 1.0 / (k - t_l - t_r)
    return float(weight * xs[t_l + 1 : k - t_r].sum())


def _s_a(x: np.ndarray, mu: float) -> float:
    return float(np.mean((x - mu) ** 2))


def _uicm(img: np.ndarray) -> float:
    """UICM with asymmetric alpha-trimmed statistics (reference protocol)."""
    # Input is BGR; the reference implementation indexes RGB.
    r = img[:, :, 2].astype(np.float64).ravel()
    g = img[:, :, 1].astype(np.float64).ravel()
    b = img[:, :, 0].astype(np.float64).ravel()
    rg = r - g
    yb = (r + g) / 2.0 - b
    mu_rg = _mu_a(rg)
    mu_yb = _mu_a(yb)
    mean_magnitude = np.sqrt(mu_rg ** 2 + mu_yb ** 2)
    r = np.sqrt(_s_a(rg, mu_rg) + _s_a(yb, mu_yb))
    return float(-0.0268 * mean_magnitude + 0.1586 * r)


def _sobel_edge_map(ch: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(ch, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(ch, cv2.CV_64F, 0, 1)
    mag = np.hypot(dx, dy)
    m = mag.max()
    if m > 0:
        mag = mag * (255.0 / m)
    return mag


def _eme(x: np.ndarray, window_size: int = 10) -> float:
    """Enhancement Measure Estimation over 10x10 blocks (reference protocol)."""
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size
    if k1 == 0 or k2 == 0:
        return 0.0
    w = 2.0 / (k1 * k2)
    x = x[: k2 * window_size, : k1 * window_size]
    val = 0.0
    for column in range(k1):
        for k in range(k2):
            block = x[k * window_size : window_size * (k + 1), column * window_size : window_size * (column + 1)]
            bmax = float(block.max())
            bmin = float(block.min())
            if bmin == 0.0 or bmax == 0.0:
                continue
            val += np.log(bmax / bmin)
    return float(w * val)


def _uism(img: np.ndarray) -> float:
    """UISM: EME of (channel-normalized Sobel edges x channel), 10x10 blocks."""
    # RGB order per the reference implementation.
    chans = (
        img[:, :, 2].astype(np.float64),
        img[:, :, 1].astype(np.float64),
        img[:, :, 0].astype(np.float64),
    )
    emes = [_eme(_sobel_edge_map(c) * c, 10) for c in chans]
    # Reference constants (lambda_r, lambda_g, lambda_b).
    return 0.299 * emes[0] + 0.587 * emes[1] + 0.144 * emes[2]


def _uiconm(img: np.ndarray, window_size: int = 10) -> float:
    """UIConM: logAMEE over 10x10 blocks on the RGB image (reference protocol)."""
    x = img.astype(np.float64)
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size
    if k1 == 0 or k2 == 0:
        return 0.0
    w = -1.0 / (k1 * k2)
    x = x[: k2 * window_size, : k1 * window_size]
    val = 0.0
    for column in range(k1):
        for k in range(k2):
            block = x[k * window_size : window_size * (k + 1), column * window_size : window_size * (column + 1), :]
            bmax = float(block.max())
            bmin = float(block.min())
            top = bmax - bmin
            bot = bmax + bmin
            if bot == 0.0 or top == 0.0:
                continue
            val += (top / bot) * np.log(top / bot)
    return float(w * val)


class UIQMModule(PipelineModule):
    name = "uiqm"
    provenance = "adapted"
    sources = {
        "uiqm_score": "UIQM, Panetta et al. IEEE JOE 2016 (FUnIE-GAN uqim_utils port) — https://ieeexplore.ieee.org/document/7305804",
    }
    deviations = {
        "uiqm_score": "The published metric is image-level; Ayase additionally averages uniformly sampled video frames, and configurable component weights can differ from the published defaults",
    }
    description = "UIQM underwater image quality measure (Panetta et al. 2016)"
    default_config = {
        "c1": 0.0282,
        "c2": 0.2953,
        "c3": 3.5753,
        "subsample": 8,
    }
    metric_groups = {
        "uiqm_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.c1 = self.config.get("c1", 0.0282)
        self.c2 = self.config.get("c2", 0.2953)
        self.c3 = self.config.get("c3", 3.5753)
        self.subsample = self.config.get("subsample", 8)
        # Faithful port of the Panetta et al. (2016) UIQM formula
        # (UICM + UISM + UIConM).
        self._backend = "port"

    def process(self, sample: Sample) -> Sample:
        try:
            score = self._compute(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.uiqm_score = score
        except Exception as e:
            logger.warning(f"UIQM failed for {sample.path}: {e}")
        return sample

    def _score_frame(self, frame: np.ndarray) -> float:
        uicm = _uicm(frame)
        uism = _uism(frame)
        uiconm = _uiconm(frame)
        return self.c1 * uicm + self.c2 * uism + self.c3 * uiconm

    def _compute(self, sample: Sample) -> Optional[float]:
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return None
            indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
            scores = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    scores.append(self._score_frame(frame))
            cap.release()
            return float(np.mean(scores)) if scores else None
        else:
            img = cv2.imread(str(sample.path))
            if img is None:
                return None
            return float(self._score_frame(img))

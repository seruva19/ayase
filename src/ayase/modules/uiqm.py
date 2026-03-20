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


def _uicm(img: np.ndarray) -> float:
    """Underwater Image Colorfulness Measure (UICM)."""
    r, g, b = img[:, :, 2].astype(float), img[:, :, 1].astype(float), img[:, :, 0].astype(float)
    rg = r - g
    yb = 0.5 * (r + g) - b

    mu_rg = np.mean(rg)
    mu_yb = np.mean(yb)
    sigma_rg = np.sqrt(np.var(rg))
    sigma_yb = np.sqrt(np.var(yb))

    return -0.0268 * np.sqrt(mu_rg ** 2 + mu_yb ** 2) + 0.1586 * np.sqrt(sigma_rg ** 2 + sigma_yb ** 2)


def _uism(img: np.ndarray) -> float:
    """Underwater Image Sharpness Measure (UISM) via Sobel edge detector."""
    # Compute edge maps per channel using Sobel
    channels = cv2.split(img)
    eme_vals = []

    for ch in channels:
        ch = ch.astype(np.float64)
        sobel_x = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
        edge_map = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Block-based EME (Measure of Enhancement)
        h, w = edge_map.shape
        block_h, block_w = max(h // 8, 1), max(w // 8, 1)
        eme = 0.0
        count = 0
        for i in range(0, h - block_h + 1, block_h):
            for j in range(0, w - block_w + 1, block_w):
                block = edge_map[i : i + block_h, j : j + block_w]
                bmax = block.max()
                bmin = block.min()
                if bmin > 0 and bmax > 0:
                    eme += 20.0 * np.log(bmax / bmin + 1e-8)
                    count += 1

        eme_vals.append(eme / max(count, 1))

    # Weighted sum (lambda_r=0.299, lambda_g=0.587, lambda_b=0.114)
    return 0.299 * eme_vals[2] + 0.587 * eme_vals[1] + 0.114 * eme_vals[0]


def _uiconm(img: np.ndarray) -> float:
    """Underwater Image Contrast Measure (UIConM) via logAMEE."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    block_h, block_w = max(h // 8, 1), max(w // 8, 1)

    logamee = 0.0
    count = 0

    for i in range(0, h - block_h + 1, block_h):
        for j in range(0, w - block_w + 1, block_w):
            block = gray[i : i + block_h, j : j + block_w]
            bmax = block.max()
            bmin = block.min()
            if bmin > 0 and bmax > bmin:
                alpha = (bmax + bmin) / 2.0
                plip_add = bmax + bmin - bmax * bmin / alpha if alpha > 0 else 0
                plip_sub = abs(bmax - bmin)
                if plip_add > 0 and plip_sub > 0:
                    logamee += np.log(plip_sub / plip_add + 1e-8)
                    count += 1

    return logamee / max(count, 1)


class UIQMModule(PipelineModule):
    name = "uiqm"
    description = "UIQM underwater image quality measure (Panetta et al. 2016)"
    default_config = {
        "c1": 0.0282,
        "c2": 0.2953,
        "c3": 3.5753,
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.c1 = self.config.get("c1", 0.0282)
        self.c2 = self.config.get("c2", 0.2953)
        self.c3 = self.config.get("c3", 3.5753)
        self.subsample = self.config.get("subsample", 8)

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

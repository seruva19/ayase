"""UCIQE — Underwater Color Image Quality Evaluation.

Yang & Sowmya 2015 — pure CIELab-based metric for underwater images.
Combines chroma standard deviation, luminance contrast, and saturation.

GitHub: https://github.com/paulwong16/UCIQE

uciqe_score — higher = better quality

Formula: UCIQE = c1*σ_c + c2*con_l + c3*μ_s
Default weights: c1=0.4680, c2=0.2745, c3=0.2576
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _compute_uciqe(img: np.ndarray, c1: float, c2: float, c3: float) -> float:
    """Compute UCIQE score for a single BGR image.

    Uses standard CIELab [0-100] range (rescaled from OpenCV's [0-255]).
    Formula: UCIQE = c1*σ_c + c2*con_l + c3*μ_s  (Yang & Sowmya 2015)
    """
    # Convert to CIELab — OpenCV returns L in [0,255], a/b in [0,255] (128-centered)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)

    # Rescale to standard CIELab ranges: L [0,100], a [-128,127], b [-128,127]
    l_channel = lab[:, :, 0] * (100.0 / 255.0)
    a_channel = lab[:, :, 1] - 128.0
    b_channel = lab[:, :, 2] - 128.0

    # Chroma
    chroma = np.sqrt(a_channel ** 2 + b_channel ** 2)
    sigma_c = np.std(chroma)  # chroma std

    # Luminance contrast: top 1% - bottom 1%
    l_sorted = np.sort(l_channel.ravel())
    n = len(l_sorted)
    top_1 = np.mean(l_sorted[int(0.99 * n) :])
    bot_1 = np.mean(l_sorted[: max(int(0.01 * n), 1)])
    con_l = top_1 - bot_1

    # Saturation: S = C / L (paper formula)
    saturation = chroma / (l_channel + 1e-8)
    mu_s = np.mean(saturation)

    return c1 * sigma_c + c2 * con_l + c3 * mu_s


class UCIQEModule(PipelineModule):
    name = "uciqe"
    description = "UCIQE underwater color image quality evaluation (2015)"
    default_config = {
        "c1": 0.4680,
        "c2": 0.2745,
        "c3": 0.2576,
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.c1 = self.config.get("c1", 0.4680)
        self.c2 = self.config.get("c2", 0.2745)
        self.c3 = self.config.get("c3", 0.2576)
        self.subsample = self.config.get("subsample", 8)

    def process(self, sample: Sample) -> Sample:
        try:
            score = self._compute(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.uciqe_score = score
        except Exception as e:
            logger.warning(f"UCIQE failed for {sample.path}: {e}")
        return sample

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
                    scores.append(_compute_uciqe(frame, self.c1, self.c2, self.c3))
            cap.release()
            return float(np.mean(scores)) if scores else None
        else:
            img = cv2.imread(str(sample.path))
            if img is None:
                return None
            return float(_compute_uciqe(img, self.c1, self.c2, self.c3))

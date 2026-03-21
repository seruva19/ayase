"""CIEDE2000 module.

CIEDE2000 is the standard perceptual color difference metric (CIE DE2000).
It measures how different two colors appear to the human eye, accounting
for luminance, chroma, and hue weighting.

Range: 0+ (lower = more similar, 0 = identical color).
  < 1: imperceptible, 1-2: barely perceptible, 2-10: noticeable, > 10: large

Full-reference metric. Pure OpenCV/numpy implementation.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


def _rescale_lab(lab: np.ndarray) -> np.ndarray:
    """Rescale OpenCV 8-bit LAB to standard CIELAB ranges.

    OpenCV 8-bit: L=[0,255], a=[0,255], b=[0,255] (a,b centered at 128)
    CIELAB:       L=[0,100], a=[-128,127], b=[-128,127]
    """
    out = lab.copy()
    out[:, :, 0] = out[:, :, 0] * (100.0 / 255.0)
    out[:, :, 1] = out[:, :, 1] - 128.0
    out[:, :, 2] = out[:, :, 2] - 128.0
    return out


def _ciede2000_pixel(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000 computation for arrays of LAB values.

    lab1, lab2: (H, W, 3) float arrays in CIE LAB space.
    Returns: (H, W) array of Delta E 2000 values.
    """
    L1, a1, b1 = lab1[:, :, 0], lab1[:, :, 1], lab1[:, :, 2]
    L2, a2, b2 = lab2[:, :, 0], lab2[:, :, 1], lab2[:, :, 2]

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_avg = (C1 + C2) / 2.0

    C_avg7 = C_avg ** 7
    G = 0.5 * (1 - np.sqrt(C_avg7 / (C_avg7 + 25.0 ** 7)))

    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)

    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = np.where(
        C1p * C2p == 0, 0,
        np.where(np.abs(h2p - h1p) <= 180, h2p - h1p,
                 np.where(h2p - h1p > 180, h2p - h1p - 360, h2p - h1p + 360))
    )
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0

    hp_avg = np.where(
        C1p * C2p == 0, h1p + h2p,
        np.where(np.abs(h1p - h2p) <= 180, (h1p + h2p) / 2.0,
                 np.where(h1p + h2p < 360, (h1p + h2p + 360) / 2.0, (h1p + h2p - 360) / 2.0))
    )

    T = (1 - 0.17 * np.cos(np.radians(hp_avg - 30))
         + 0.24 * np.cos(np.radians(2 * hp_avg))
         + 0.32 * np.cos(np.radians(3 * hp_avg + 6))
         - 0.20 * np.cos(np.radians(4 * hp_avg - 63)))

    SL = 1 + 0.015 * (Lp_avg - 50) ** 2 / np.sqrt(20 + (Lp_avg - 50) ** 2)
    SC = 1 + 0.045 * Cp_avg
    SH = 1 + 0.015 * Cp_avg * T

    Cp_avg7 = Cp_avg ** 7
    RT = -2 * np.sqrt(Cp_avg7 / (Cp_avg7 + 25.0 ** 7)) * np.sin(
        np.radians(60 * np.exp(-((hp_avg - 275) / 25.0) ** 2))
    )

    dE = np.sqrt(np.maximum(0,
        (dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    ))
    return dE


class CIEDE2000Module(ReferenceBasedModule):
    name = "ciede2000"
    description = "CIEDE2000 perceptual color difference (lower=better)"
    default_config = {"subsample": 5}

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self._ml_available = True  # Pure numpy

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        try:
            ref = cv2.imread(str(reference_path))
            dist = cv2.imread(str(sample_path))
            if ref is None or dist is None:
                return None

            h = min(ref.shape[0], dist.shape[0])
            w = min(ref.shape[1], dist.shape[1])
            ref = cv2.resize(ref, (w, h))
            dist = cv2.resize(dist, (w, h))

            ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
            dist_lab = cv2.cvtColor(dist, cv2.COLOR_BGR2LAB).astype(np.float32)

            # OpenCV 8-bit LAB: L=[0,255], a=[0,255], b=[0,255] (centered at 128)
            # CIEDE2000 expects: L=[0,100], a=[-128,127], b=[-128,127]
            ref_lab = _rescale_lab(ref_lab)
            dist_lab = _rescale_lab(dist_lab)

            de = _ciede2000_pixel(ref_lab, dist_lab)
            return float(np.mean(de))
        except Exception as e:
            logger.debug(f"CIEDE2000 failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        reference = Path(reference) if not isinstance(reference, Path) else reference
        if not reference.exists():
            return sample

        try:
            if sample.is_video:
                score = self._process_video(sample.path, reference)
            else:
                score = self.compute_reference_score(sample.path, reference)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.ciede2000 = score
                logger.debug(f"CIEDE2000 for {sample.path.name}: {score:.2f}")
        except Exception as e:
            logger.error(f"CIEDE2000 failed: {e}")
        return sample

    def _process_video(self, path, ref_path) -> Optional[float]:
        ref_cap = cv2.VideoCapture(str(ref_path))
        dist_cap = cv2.VideoCapture(str(path))
        scores = []
        idx = 0
        while True:
            r1, rf = ref_cap.read()
            r2, df = dist_cap.read()
            if not r1 or not r2:
                break
            if idx % self.subsample == 0:
                h = min(rf.shape[0], df.shape[0])
                w = min(rf.shape[1], df.shape[1])
                rf = cv2.resize(rf, (w, h))
                df = cv2.resize(df, (w, h))
                ref_lab = _rescale_lab(cv2.cvtColor(rf, cv2.COLOR_BGR2LAB).astype(np.float32))
                dist_lab = _rescale_lab(cv2.cvtColor(df, cv2.COLOR_BGR2LAB).astype(np.float32))
                scores.append(float(np.mean(_ciede2000_pixel(ref_lab, dist_lab))))
            idx += 1
        ref_cap.release()
        dist_cap.release()
        return float(np.mean(scores)) if scores else None

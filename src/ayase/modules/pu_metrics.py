"""PU-PSNR and PU-SSIM (Perceptually Uniform HDR) module.

PU21 encoding transforms absolute linear RGB values to be perceptually
uniform, enabling standard metrics like PSNR/SSIM to work on HDR content.

PU-PSNR: dB scale (higher = better)
PU-SSIM: 0-1 (higher = better)

Full-reference metrics for HDR content.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


def _pu21_encode(linear_rgb: np.ndarray) -> np.ndarray:
    """Apply PU21 perceptual uniform encoding to linear RGB values.

    Maps absolute luminance (nits) to perceptually uniform values.
    Simplified PU21 transfer function.
    """
    # Clamp to avoid log(0)
    L = np.clip(linear_rgb, 1e-6, None)
    # PU21 approximation (simplified Barten CSF-based)
    # Based on: V = a * log(1 + b * L) / log(1 + b * L_max)
    a = 0.1703
    b = 2.2290
    L_max = 10000.0  # Peak luminance in nits
    pu = a * np.log(1.0 + b * L) / np.log(1.0 + b * L_max)
    return pu.astype(np.float32)


class PUMetricsModule(ReferenceBasedModule):
    name = "pu_metrics"
    description = "PU-PSNR + PU-SSIM for HDR content (perceptually uniform)"
    default_config = {
        "subsample": 5,
        "assume_nits_range": 10000.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self.nits_range = self.config.get("assume_nits_range", 10000.0)
        self._ml_available = True  # Pure numpy, always available

    def setup(self) -> None:
        logger.info("PU metrics module initialised (numpy-based)")

    def _compute_pu_psnr(self, ref: np.ndarray, dist: np.ndarray) -> float:
        ref_pu = _pu21_encode(ref.astype(np.float32))
        dist_pu = _pu21_encode(dist.astype(np.float32))
        mse = float(np.mean((ref_pu - dist_pu) ** 2))
        if mse < 1e-10:
            return 100.0
        max_val = float(np.max(ref_pu))
        if max_val < 1e-10:
            max_val = 1.0
        return float(10.0 * np.log10(max_val ** 2 / mse))

    def _compute_pu_ssim(self, ref: np.ndarray, dist: np.ndarray) -> float:
        ref_pu = _pu21_encode(ref.astype(np.float32))
        dist_pu = _pu21_encode(dist.astype(np.float32))

        # Convert to single-channel grayscale for SSIM
        if ref_pu.ndim == 3:
            ref_gray = np.mean(ref_pu, axis=2)
            dist_gray = np.mean(dist_pu, axis=2)
        else:
            ref_gray = ref_pu
            dist_gray = dist_pu

        # Simplified SSIM on PU-encoded values
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2

        mu_ref = cv2.GaussianBlur(ref_gray, (11, 11), 1.5)
        mu_dist = cv2.GaussianBlur(dist_gray, (11, 11), 1.5)
        mu_ref_sq = mu_ref ** 2
        mu_dist_sq = mu_dist ** 2
        mu_ref_dist = mu_ref * mu_dist

        sigma_ref = cv2.GaussianBlur(ref_gray ** 2, (11, 11), 1.5) - mu_ref_sq
        sigma_dist = cv2.GaussianBlur(dist_gray ** 2, (11, 11), 1.5) - mu_dist_sq
        sigma_cross = cv2.GaussianBlur(ref_gray * dist_gray, (11, 11), 1.5) - mu_ref_dist

        num = (2 * mu_ref_dist + C1) * (2 * sigma_cross + C2)
        den = (mu_ref_sq + mu_dist_sq + C1) * (sigma_ref + sigma_dist + C2)

        ssim_map = num / (den + 1e-10)
        return float(np.clip(np.mean(ssim_map), 0, 1))

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        # Not used directly; process() handles both metrics
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
                psnr_val, ssim_val = self._process_video(sample.path, reference)
            else:
                ref_img = cv2.imread(str(reference))
                dist_img = cv2.imread(str(sample.path))
                if ref_img is None or dist_img is None:
                    return sample
                ref_img = ref_img.astype(np.float32)
                dist_img = dist_img.astype(np.float32)
                h = min(ref_img.shape[0], dist_img.shape[0])
                w = min(ref_img.shape[1], dist_img.shape[1])
                ref_img = cv2.resize(ref_img, (w, h))
                dist_img = cv2.resize(dist_img, (w, h))
                # Scale to nits range
                ref_img = ref_img / 255.0 * self.nits_range
                dist_img = dist_img / 255.0 * self.nits_range
                psnr_val = self._compute_pu_psnr(ref_img, dist_img)
                ssim_val = self._compute_pu_ssim(ref_img, dist_img)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            if psnr_val is not None:
                sample.quality_metrics.pu_psnr = psnr_val
            if ssim_val is not None:
                sample.quality_metrics.pu_ssim = ssim_val
        except Exception as e:
            logger.error(f"PU metrics failed: {e}")
        return sample

    def _process_video(self, path, ref_path):
        ref_cap = cv2.VideoCapture(str(ref_path))
        dist_cap = cv2.VideoCapture(str(path))
        psnrs, ssims = [], []
        idx = 0
        while True:
            r1, ref_f = ref_cap.read()
            r2, dist_f = dist_cap.read()
            if not r1 or not r2:
                break
            if idx % self.subsample == 0:
                ref_f = ref_f.astype(np.float32) / 255.0 * self.nits_range
                dist_f = dist_f.astype(np.float32) / 255.0 * self.nits_range
                h = min(ref_f.shape[0], dist_f.shape[0])
                w = min(ref_f.shape[1], dist_f.shape[1])
                ref_f = cv2.resize(ref_f, (w, h))
                dist_f = cv2.resize(dist_f, (w, h))
                psnrs.append(self._compute_pu_psnr(ref_f, dist_f))
                ssims.append(self._compute_pu_ssim(ref_f, dist_f))
            idx += 1
        ref_cap.release()
        dist_cap.release()
        p = float(np.mean(psnrs)) if psnrs else None
        s = float(np.mean(ssims)) if ssims else None
        return p, s

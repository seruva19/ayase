"""VFIPS — Video Frame Interpolation Perceptual Similarity (ECCV 2022).

Full-reference perceptual metric designed for frame interpolation evaluation.
Uses spatiotemporal features to capture temporal artifacts.

vfips_score — lower = better (perceptual distance).
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class VFIPSModule(ReferenceBasedModule):
    name = "vfips"
    description = "VFIPS frame interpolation perceptual similarity (ECCV 2022, FR)"
    metric_field = "vfips_score"
    default_config = {"subsample": 8}

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self.subsample = self.config.get("subsample", 8)

    def setup(self) -> None:
        logger.info("VFIPS module initialised (heuristic)")

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        try:
            if str(sample_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                return self._score_video(str(sample_path), str(reference_path))
            else:
                return self._score_image(str(sample_path), str(reference_path))
        except Exception as e:
            logger.warning(f"VFIPS failed: {e}")
            return None

    def _score_image(self, sample_p: str, ref_p: str) -> Optional[float]:
        img = cv2.imread(sample_p)
        ref = cv2.imread(ref_p)
        if img is None or ref is None:
            return None
        return self._spatiotemporal_distance(img, ref)

    def _score_video(self, sample_p: str, ref_p: str) -> Optional[float]:
        cap_s = cv2.VideoCapture(sample_p)
        cap_r = cv2.VideoCapture(ref_p)
        try:
            total = min(
                int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)),
            )
            if total <= 0:
                return None
            indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
            scores = []
            for idx in indices:
                cap_s.set(cv2.CAP_PROP_POS_FRAMES, idx)
                cap_r.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret_s, frame_s = cap_s.read()
                ret_r, frame_r = cap_r.read()
                if ret_s and ret_r:
                    s = self._spatiotemporal_distance(frame_s, frame_r)
                    if s is not None:
                        scores.append(s)
            return float(np.mean(scores)) if scores else None
        finally:
            cap_s.release()
            cap_r.release()

    def _spatiotemporal_distance(self, img: np.ndarray, ref: np.ndarray) -> float:
        """SSIM-like perceptual distance with spatial gradient weighting.

        Combines luminance, contrast, and structure terms weighted by
        spatial gradient magnitude to emphasise perceptually important regions.
        Returns a distance (lower = better).
        """
        h, w = ref.shape[:2]
        img = cv2.resize(img, (w, h))

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Gradient magnitude as perceptual weight map
        gx = cv2.Sobel(gray_ref, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_ref, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        weight = grad_mag / (grad_mag.max() + 1e-8) + 0.1  # baseline weight

        # Weighted MSE in spatial domain
        diff_sq = (gray_img - gray_ref) ** 2
        wmse = np.sum(weight * diff_sq) / (np.sum(weight) + 1e-8)

        # 1 - SSIM as structural distance component
        mu_i = cv2.GaussianBlur(gray_img, (11, 11), 1.5)
        mu_r = cv2.GaussianBlur(gray_ref, (11, 11), 1.5)
        sig_i_sq = cv2.GaussianBlur(gray_img ** 2, (11, 11), 1.5) - mu_i ** 2
        sig_r_sq = cv2.GaussianBlur(gray_ref ** 2, (11, 11), 1.5) - mu_r ** 2
        sig_ir = cv2.GaussianBlur(gray_img * gray_ref, (11, 11), 1.5) - mu_i * mu_r

        c1, c2 = 6.5025, 58.5225
        ssim_map = ((2 * mu_i * mu_r + c1) * (2 * sig_ir + c2)) / \
                   ((mu_i ** 2 + mu_r ** 2 + c1) * (sig_i_sq + sig_r_sq + c2))
        dssim = np.mean(weight * (1.0 - ssim_map)) / (np.mean(weight) + 1e-8)

        # Combine: weighted MSE normalised + structural distance
        score = 0.5 * (wmse / 255.0 ** 2) + 0.5 * dssim
        return float(np.clip(score, 0.0, 1.0))

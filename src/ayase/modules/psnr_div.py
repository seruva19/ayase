"""PSNR_DIV — Motion-Weighted PSNR for Frame Interpolation (ICIP 2025).

Full-reference metric that weights PSNR by motion-field divergence,
giving more importance to regions with complex motion where interpolation
artefacts are most visible.

psnr_div — dB, higher = better.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class PSNRDIVModule(ReferenceBasedModule):
    name = "psnr_div"
    description = "PSNR_DIV motion-weighted PSNR for frame interpolation (ICIP 2025, FR)"
    metric_field = "psnr_div"
    default_config = {"subsample": 8, "block_size": 16}

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self.subsample = self.config.get("subsample", 8)
        self.block_size = self.config.get("block_size", 16)

    def setup(self) -> None:
        logger.info("PSNR_DIV module initialised (heuristic)")

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        try:
            if str(sample_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                return self._score_video(str(sample_path), str(reference_path))
            else:
                return self._score_image(str(sample_path), str(reference_path))
        except Exception as e:
            logger.warning(f"PSNR_DIV failed: {e}")
            return None

    def _score_image(self, sample_p: str, ref_p: str) -> Optional[float]:
        img = cv2.imread(sample_p)
        ref = cv2.imread(ref_p)
        if img is None or ref is None:
            return None
        return self._motion_weighted_psnr(img, ref)

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
                    s = self._motion_weighted_psnr(frame_s, frame_r)
                    if s is not None:
                        scores.append(s)
            return float(np.mean(scores)) if scores else None
        finally:
            cap_s.release()
            cap_r.release()

    def _motion_weighted_psnr(self, img: np.ndarray, ref: np.ndarray) -> Optional[float]:
        """Compute motion-divergence-weighted PSNR.

        Blocks in regions with high gradient divergence (proxy for motion
        complexity) receive higher weight.
        """
        h, w = ref.shape[:2]
        img = cv2.resize(img, (w, h))

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Estimate motion divergence proxy from gradient differences
        gx_img = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        gy_img = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        gx_ref = cv2.Sobel(gray_ref, cv2.CV_64F, 1, 0, ksize=3)
        gy_ref = cv2.Sobel(gray_ref, cv2.CV_64F, 0, 1, ksize=3)

        # Divergence proxy: magnitude of gradient difference
        div_map = np.sqrt((gx_img - gx_ref) ** 2 + (gy_img - gy_ref) ** 2)

        bs = self.block_size
        weighted_mse = 0.0
        total_weight = 0.0

        for y in range(0, h - bs + 1, bs):
            for x in range(0, w - bs + 1, bs):
                block_img = gray_img[y:y + bs, x:x + bs]
                block_ref = gray_ref[y:y + bs, x:x + bs]
                block_div = div_map[y:y + bs, x:x + bs]

                mse = np.mean((block_img - block_ref) ** 2)
                # Weight: 1 + normalised divergence (higher divergence = more weight)
                weight = 1.0 + np.mean(block_div) / 255.0
                weighted_mse += weight * mse
                total_weight += weight

        if total_weight < 1e-8:
            return None

        wmse = weighted_mse / total_weight
        if wmse < 1e-10:
            return 100.0  # perfect match
        psnr = 10.0 * np.log10(255.0 ** 2 / wmse)
        return float(np.clip(psnr, 0.0, 100.0))

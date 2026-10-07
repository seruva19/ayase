"""PSNR99 — Image-Difficulty-Aware Evaluation for Super-Resolution (2025).

Full-reference metric from "Image-Difficulty-Aware Evaluation of SR Models"
(arXiv 2509.26398): the PSNR derived from the mean of the top-1% largest
per-pixel squared errors on the luma (Y) channel — the worst-1%-pixel PSNR.

psnr99 — dB, higher = better.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class PSNR99Module(ReferenceBasedModule):
    name = "psnr99"
    provenance = "adapted"
    sources = {
        "psnr99": "PSNR99 (Image-Difficulty-Aware Evaluation of SR Models, arXiv 2509.26398) — https://arxiv.org/abs/2509.26398",
    }
    deviations = {
        "psnr99": "Image inputs use the published per-image formula. Video inputs use "
        "Ayase-defined uniform frame subsampling and mean aggregation; the shared "
        "image/video field is classified as adapted.",
    }
    description = "PSNR99 worst-1%-pixel luma PSNR for super-resolution (FR, 2025)"
    metric_field = "psnr99"
    default_config = {"subsample": 8}
    metric_groups = {
        "psnr99": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self.subsample = self.config.get("subsample", 8)
        self._backend = "numpy"

    def setup(self) -> None:
        logger.info("PSNR99 module initialised (worst-case block-PSNR, numpy)")

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        try:
            if str(sample_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                return self._score_video(str(sample_path), str(reference_path))
            else:
                return self._score_image(str(sample_path), str(reference_path))
        except Exception as e:
            logger.warning(f"PSNR99 failed: {e}")
            return None

    def _score_image(self, sample_p: str, ref_p: str) -> Optional[float]:
        img = cv2.imread(sample_p)
        ref = cv2.imread(ref_p)
        if img is None or ref is None:
            return None
        return self._block_psnr99(img, ref)

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
                    s = self._block_psnr99(frame_s, frame_r)
                    if s is not None:
                        scores.append(s)
            return float(np.mean(scores)) if scores else None
        finally:
            cap_s.release()
            cap_r.release()

    def _block_psnr99(self, img: np.ndarray, ref: np.ndarray) -> Optional[float]:
        """Mean of the top-1% per-pixel squared Y errors → PSNR (the paper's def)."""
        h, w = ref.shape[:2]
        img = cv2.resize(img, (w, h))

        y_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_IYUV)[..., 0].astype(np.float64)
        y_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2YUV_IYUV)[..., 0].astype(np.float64)

        sq_err = (y_img - y_ref) ** 2
        n_top = max(1, int(np.ceil(sq_err.size * 0.01)))
        # Mean of the largest 1% of per-pixel squared errors.
        top = np.partition(sq_err.reshape(-1), sq_err.size - n_top)[-n_top:]
        mse_top1 = float(np.mean(top))

        if mse_top1 < 1e-10:
            return 100.0  # identical inputs: unbounded PSNR, keep finite
        return float(10.0 * np.log10(255.0 ** 2 / mse_top1))

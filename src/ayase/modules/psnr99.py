"""PSNR99 — Worst-Case Region PSNR for Super-Resolution (2025).

Full-reference metric that computes per-block MSE and uses the 99th
percentile (worst blocks) to derive a PSNR score, capturing localised
quality drops missed by average PSNR.

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
    description = "PSNR99 worst-case region quality for super-resolution (FR, 2025)"
    metric_field = "psnr99"
    default_config = {"subsample": 8, "block_size": 32}

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self.subsample = self.config.get("subsample", 8)
        self.block_size = self.config.get("block_size", 32)

    def setup(self) -> None:
        logger.info("PSNR99 module initialised (heuristic)")

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
        """Compute per-block MSE, take 99th percentile, convert to PSNR."""
        h, w = ref.shape[:2]
        img = cv2.resize(img, (w, h))

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float64)

        bs = self.block_size
        block_mses = []
        for y in range(0, h - bs + 1, bs):
            for x in range(0, w - bs + 1, bs):
                block_img = gray_img[y:y + bs, x:x + bs]
                block_ref = gray_ref[y:y + bs, x:x + bs]
                mse = np.mean((block_img - block_ref) ** 2)
                block_mses.append(mse)

        if not block_mses:
            return None

        # 99th percentile MSE (worst 1% of blocks)
        mse_99 = float(np.percentile(block_mses, 99))

        if mse_99 < 1e-10:
            return 100.0  # perfect match
        psnr = 10.0 * np.log10(255.0 ** 2 / mse_99)
        return float(np.clip(psnr, 0.0, 100.0))

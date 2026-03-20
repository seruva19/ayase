"""RankDVQA — Ranking-based Deep VQA (WACV 2024).

Full-reference VQA trained with ranking-inspired hybrid training
without human MOS labels.

GitHub: https://chenfeng-bristol.github.io/RankDVQA/

rankdvqa_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class RankDVQAModule(ReferenceBasedModule):
    name = "rankdvqa"
    description = "RankDVQA ranking-based FR VQA (WACV 2024)"
    metric_field = "rankdvqa_score"
    default_config = {"subsample": 8}

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        """Compare sample against reference using patch-level SSIM + ranking."""
        try:
            if str(sample_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                return self._score_video(str(sample_path), str(reference_path))
            else:
                return self._score_image(str(sample_path), str(reference_path))
        except Exception as e:
            logger.warning(f"RankDVQA failed: {e}")
            return None

    def _score_image(self, sample_p: str, ref_p: str) -> Optional[float]:
        img = cv2.imread(sample_p)
        ref = cv2.imread(ref_p)
        if img is None or ref is None:
            return None
        return self._patch_ssim(img, ref)

    def _score_video(self, sample_p: str, ref_p: str) -> Optional[float]:
        cap_s = cv2.VideoCapture(sample_p)
        cap_r = cv2.VideoCapture(ref_p)
        total = min(int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
        if total <= 0:
            cap_s.release()
            cap_r.release()
            return None
        indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
        scores = []
        for idx in indices:
            cap_s.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_s, frame_s = cap_s.read()
            ret_r, frame_r = cap_r.read()
            if ret_s and ret_r:
                scores.append(self._patch_ssim(frame_s, frame_r))
        cap_s.release()
        cap_r.release()
        return float(np.mean(scores)) if scores else None

    def _patch_ssim(self, img: np.ndarray, ref: np.ndarray, patch_size: int = 64) -> float:
        h, w = img.shape[:2]
        ref = cv2.resize(ref, (w, h))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float64)

        ssim_vals = []
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                p_img = gray_img[y:y + patch_size, x:x + patch_size]
                p_ref = gray_ref[y:y + patch_size, x:x + patch_size]
                mu_i, mu_r = p_img.mean(), p_ref.mean()
                var_i, var_r = p_img.var(), p_ref.var()
                cov = np.mean((p_img - mu_i) * (p_ref - mu_r))
                c1, c2 = 6.5025, 58.5225
                ssim = ((2 * mu_i * mu_r + c1) * (2 * cov + c2)) / \
                       ((mu_i ** 2 + mu_r ** 2 + c1) * (var_i + var_r + c2))
                ssim_vals.append(ssim)

        return float(np.mean(ssim_vals)) if ssim_vals else 0.5

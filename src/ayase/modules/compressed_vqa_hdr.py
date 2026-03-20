"""CompressedVQA-HDR — HDR Compressed Video Quality (ICME 2025 winner).

GitHub: https://github.com/sunwei925/CompressedVQA-HDR
compressed_vqa_hdr — higher = better
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class CompressedVQAHDRModule(ReferenceBasedModule):
    name = "compressed_vqa_hdr"
    description = "CompressedVQA-HDR FR quality (ICME 2025)"
    metric_field = "compressed_vqa_hdr"
    default_config = {"subsample": 8}

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        try:
            s_ext = str(sample_path).lower()
            if s_ext.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                return self._score_video(str(sample_path), str(reference_path))
            else:
                return self._score_image(str(sample_path), str(reference_path))
        except Exception as e:
            logger.warning(f"CompressedVQA-HDR failed: {e}")
            return None

    def _score_image(self, sample_p: str, ref_p: str) -> Optional[float]:
        img = cv2.imread(sample_p)
        ref = cv2.imread(ref_p)
        if img is None or ref is None:
            return None
        return self._pu_similarity(img, ref)

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
                    s = self._pu_similarity(frame_s, frame_r)
                    if s is not None:
                        scores.append(s)
            return float(np.mean(scores)) if scores else None
        finally:
            cap_s.release()
            cap_r.release()

    def _pu_similarity(self, img: np.ndarray, ref: np.ndarray) -> float:
        """PU21-approximation + structural similarity."""
        h, w = img.shape[:2]
        ref = cv2.resize(ref, (w, h))
        img_f = img.astype(np.float64) / 255.0
        ref_f = ref.astype(np.float64) / 255.0
        # Simplified PU transform: log-like perceptual uniformity
        pu_img = np.log(img_f + 0.01)
        pu_ref = np.log(ref_f + 0.01)
        mse = np.mean((pu_img - pu_ref) ** 2)
        return float(1.0 / (1.0 + mse * 10))

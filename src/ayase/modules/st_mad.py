"""ST-MAD — Spatiotemporal Most Apparent Distortion (TIP 2012).

Full-reference metric using spatiotemporal slices + contrast masking.
st_mad — lower = better (distortion magnitude)
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class STMADModule(ReferenceBasedModule):
    name = "st_mad"
    description = "ST-MAD spatiotemporal MAD (TIP 2012)"
    metric_field = "st_mad"
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
            logger.warning(f"ST-MAD failed: {e}")
            return None

    def _score_image(self, sample_p: str, ref_p: str) -> Optional[float]:
        img = cv2.imread(sample_p, cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
        if img is None or ref is None:
            return None
        return self._spatial_mad(img, ref)

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
            spatial_mads = []
            prev_s, prev_r = None, None

            for idx in indices:
                cap_s.set(cv2.CAP_PROP_POS_FRAMES, idx)
                cap_r.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret_s, frame_s = cap_s.read()
                ret_r, frame_r = cap_r.read()
                if not (ret_s and ret_r):
                    continue

                gray_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

                # Spatial MAD per frame
                spatial_mads.append(self._spatial_mad(gray_s, gray_r))

                # Temporal MAD: difference of frame diffs
                if prev_s is not None:
                    diff_s = np.abs(gray_s.astype(float) - prev_s.astype(float))
                    diff_r = np.abs(gray_r.astype(float) - prev_r.astype(float))
                    temporal_err = np.mean(np.abs(diff_s - diff_r))
                    spatial_mads[-1] = 0.7 * spatial_mads[-1] + 0.3 * temporal_err

                prev_s, prev_r = gray_s, gray_r

            return float(np.mean(spatial_mads)) if spatial_mads else None
        finally:
            cap_s.release()
            cap_r.release()

    def _spatial_mad(self, img: np.ndarray, ref: np.ndarray) -> float:
        """Visibility-weighted spatial MAD with contrast masking."""
        h, w = img.shape[:2]
        ref = cv2.resize(ref, (w, h))
        img_f = img.astype(np.float64)
        ref_f = ref.astype(np.float64)
        diff = np.abs(img_f - ref_f)
        # Contrast masking
        ref_contrast = cv2.Laplacian(ref_f, cv2.CV_64F)
        mask = 1.0 / (1.0 + np.abs(ref_contrast) * 0.01)
        return float(np.mean(diff * mask))

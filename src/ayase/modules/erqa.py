"""ERQA — Edge Restoration Quality Assessment (2022).

Full-reference metric that evaluates edge restoration quality by comparing
edge maps between distorted and reference images/frames.

pip install erqa

erqa_score — 0-1, higher = better edge restoration quality.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class ERQAModule(ReferenceBasedModule):
    name = "erqa"
    description = "ERQA edge restoration quality assessment (FR, 2022)"
    metric_field = "erqa_score"
    default_config = {"subsample": 8}
    metric_groups = {
        "erqa_score": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._ml_available = False
        self._backend = "unavailable"
        self.subsample = self.config.get("subsample", 8)

    def setup(self) -> None:
        # Real backend: the ``erqa`` package (the reference ERQA implementation).
        try:
            import erqa as erqa_lib
            self._model = erqa_lib.ERQA()
            self._ml_available = True
            self._backend = "erqa"
            logger.info("ERQA module initialised (erqa package)")
            return
        except ImportError:
            logger.warning("ERQA unavailable: `pip install erqa`. erqa_score left unset.")
        except Exception as e:
            logger.warning("ERQA unavailable: %s. erqa_score left unset.", e)

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        if not self._ml_available or self._model is None:
            return None
        try:
            if str(sample_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                return self._score_video(str(sample_path), str(reference_path))
            else:
                return self._score_image(str(sample_path), str(reference_path))
        except Exception as e:
            logger.warning(f"ERQA failed: {e}")
            return None

    def _score_image(self, sample_p: str, ref_p: str) -> Optional[float]:
        img = cv2.imread(sample_p)
        ref = cv2.imread(ref_p)
        if img is None or ref is None:
            return None
        return self._score_erqa_package(img, ref)

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
                    s = self._score_erqa_package(frame_s, frame_r)
                    if s is not None:
                        scores.append(s)
            return float(np.mean(scores)) if scores else None
        finally:
            cap_s.release()
            cap_r.release()

    def _score_erqa_package(self, img: np.ndarray, ref: np.ndarray) -> Optional[float]:
        try:
            h, w = ref.shape[:2]
            img = cv2.resize(img, (w, h))
            score = self._model(img, ref)
            return float(score)
        except Exception as e:
            logger.debug(f"ERQA package scoring failed: {e}")
            return None

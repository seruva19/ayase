"""Tonal Dynamic Range module.

Computes the luminance histogram percentile span (p1–p99) of an image
or video, normalised to 0–100.  A score of 100 means the full tonal
range is utilised; a low score indicates a flat / washed-out image.

For video inputs, ``subsample`` uniformly-spaced frames are scored and
the results averaged.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TonalDynamicRangeModule(PipelineModule):
    name = "tonal_dynamic_range"
    description = "Luminance histogram tonal range (0-100)"
    default_config = {
        "low_percentile": 1,
        "high_percentile": 99,
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.low_pct = self.config.get("low_percentile", 1)
        self.high_pct = self.config.get("high_percentile", 99)
        self.subsample = self.config.get("subsample", 8)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            if sample.is_video:
                score = self._score_video(sample)
            else:
                frame = cv2.imread(str(sample.path))
                score = self._score_frame(frame) if frame is not None else None

            if score is not None:
                sample.quality_metrics.tonal_dynamic_range = score
        except Exception as e:
            logger.warning(f"Tonal dynamic range failed for {sample.path}: {e}")

        return sample

    def _score_frame(self, frame_bgr: np.ndarray) -> Optional[float]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lo = float(np.percentile(gray, self.low_pct))
        hi = float(np.percentile(gray, self.high_pct))
        return (hi - lo) / 255.0 * 100.0

    def _score_video(self, sample: Sample) -> Optional[float]:
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None

        indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
        scores = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                s = self._score_frame(frame)
                if s is not None:
                    scores.append(s)
        cap.release()

        return float(np.mean(scores)) if scores else None

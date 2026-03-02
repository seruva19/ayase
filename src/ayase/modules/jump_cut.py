"""Jump cut detection module.

From Open-Sora 2.0 pipeline. Detects abrupt scene transitions
within a clip that indicate editing cuts.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class JumpCutModule(PipelineModule):
    name = "jump_cut"
    description = "Jump cut / abrupt transition detection (0-1, 1=no cuts)"
    default_config = {"threshold": 40.0}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not sample.is_video:
            return sample

        try:
            import cv2

            cap = cv2.VideoCapture(str(sample.path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            threshold = self.config.get("threshold", 40.0)

            # Sample densely for cut detection (every ~3 frames)
            step = max(1, int(fps / 10))
            prev_hist = None
            jump_count = 0
            frame_pairs = 0

            for i in range(0, total, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                hist = hist.flatten() / (hist.sum() + 1e-8)

                if prev_hist is not None:
                    # Chi-squared distance between histograms
                    chi2 = float(np.sum((hist - prev_hist) ** 2 / (hist + prev_hist + 1e-8)))
                    if chi2 > threshold:
                        jump_count += 1
                    frame_pairs += 1

                prev_hist = hist
            cap.release()

            if frame_pairs == 0:
                sample.quality_metrics.jump_cut_score = 1.0
            else:
                duration = total / fps
                cuts_per_second = jump_count / max(duration, 0.1)
                sample.quality_metrics.jump_cut_score = 1.0 / (1.0 + cuts_per_second * 5.0)
        except Exception as e:
            logger.warning("Jump cut detection failed: %s", e)
        return sample

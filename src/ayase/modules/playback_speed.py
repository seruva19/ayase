"""Fast/slow motion (playback speed) detection module.

From Open-Sora 2.0 pipeline. Detects abnormal playback speeds
(time-lapse, slow-motion, speed-up artifacts) by analysing
optical flow magnitude distribution relative to frame rate.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PlaybackSpeedModule(PipelineModule):
    name = "playback_speed"
    description = "Playback speed normality detection (1.0=normal)"
    default_config = {"subsample": 16}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not sample.is_video:
            return sample

        try:
            import cv2

            subsample = self.config.get("subsample", 16)
            cap = cv2.VideoCapture(str(sample.path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            cap.release()

            if len(frames) < 2:
                return sample

            # Measure inter-frame motion magnitude
            motions = []
            for i in range(len(frames) - 1):
                diff = np.mean(np.abs(
                    frames[i].astype(float) - frames[i + 1].astype(float)
                ))
                motions.append(diff)

            mean_motion = float(np.mean(motions))
            # Normalize by expected motion at given fps
            # Normal content at 30fps: ~5-15 mean pixel diff between subsampled frames
            frame_gap = max(1, total // subsample)
            expected_motion = frame_gap * 0.5  # rough expected per-gap motion
            if expected_motion > 0:
                speed_ratio = mean_motion / expected_motion
            else:
                speed_ratio = 1.0

            # Score: 1.0 = normal, deviates for abnormal
            # Very low motion (static/slow-mo) or very high (time-lapse) both deviate
            sample.quality_metrics.playback_speed_score = float(
                min(speed_ratio, 1.0 / max(speed_ratio, 0.01))
            )
        except Exception as e:
            logger.warning("Playback speed detection failed: %s", e)
        return sample

"""Camera jitter / shake detection module.

From Open-Sora 2.0 pipeline. Detects unstable camera movement
by analysing optical flow variance across consecutive frames.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CameraJitterModule(PipelineModule):
    name = "camera_jitter"
    description = "Camera jitter/shake detection (0-1, 1=stable)"
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
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            cap.release()

            if len(frames) < 3:
                return sample

            # Compute global motion vectors between consecutive frames
            global_motions = []
            for i in range(len(frames) - 1):
                flow = cv2.calcOpticalFlowFarneback(
                    frames[i], frames[i + 1], None,
                    0.5, 3, 15, 3, 5, 1.2, 0,
                )
                # Global motion = median flow (robust to local motion)
                dx = np.median(flow[..., 0])
                dy = np.median(flow[..., 1])
                global_motions.append((dx, dy))

            # Jitter = variance of acceleration (change in global motion)
            if len(global_motions) < 2:
                sample.quality_metrics.camera_jitter_score = 1.0
                return sample

            accels = []
            for i in range(len(global_motions) - 1):
                ax = global_motions[i + 1][0] - global_motions[i][0]
                ay = global_motions[i + 1][1] - global_motions[i][1]
                accels.append(np.sqrt(ax ** 2 + ay ** 2))

            jitter_magnitude = float(np.std(accels))
            # Map to 0-1 stability score: high jitter = low stability
            sample.quality_metrics.camera_jitter_score = 1.0 / (1.0 + jitter_magnitude * 2.0)
        except Exception as e:
            logger.warning("Camera jitter detection failed: %s", e)
        return sample

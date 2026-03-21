import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class TemporalStyleModule(PipelineModule):
    name = "temporal_style"
    description = "Analyzes temporal style (Slow Motion, Timelapse, Speed)"
    default_config = {}

    def __init__(self, config=None):
        super().__init__(config)

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            style = self._analyze_temporal_style(sample)
            if style:
                 sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Temporal Style: {style}",
                        details={"style": style}
                    )
                )
        except Exception as e:
            logger.warning(f"Temporal style check failed: {e}")

        return sample

    def _analyze_temporal_style(self, sample: Sample) -> str:
        max_frames = self.config.get("max_frames", 300)
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return "unknown"

        prev_gray = None
        flow_mags = []
        sampled = 0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample every 5th frame to save time
            if frame_idx % 5 != 0:
                frame_idx += 1
                continue

            sampled += 1
            if sampled > max_frames:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_mags.append(np.mean(mag))

            prev_gray = gray
            frame_idx += 1

        cap.release()
        
        if not flow_mags:
            return "static"
            
        avg_flow = np.mean(flow_mags)
        std_flow = np.std(flow_mags)
        
        # Heuristics
        if avg_flow < 0.5:
            return "static"
        elif avg_flow < 2.0:
            return "slow_motion"
        elif avg_flow > 15.0 and std_flow > 5.0:
            return "timelapse_or_fast"
        else:
            return "normal_speed"

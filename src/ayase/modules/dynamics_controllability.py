"""Dynamics Controllability module.

Assesses if motion in video matches the dynamics implied by the text prompt.
Parses caption for motion keywords and compares to actual motion.
Range: 0-1 (higher = better controllability/alignment).
"""

import logging
import re

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DynamicsControllabilityModule(PipelineModule):
    name = "dynamics_controllability"
    description = "Assesses motion controllability based on text-motion alignment"
    default_config = {
        # Rule-based + OpenCV
    }

    # Motion keywords and their expected motion levels (0-1)
    MOTION_KEYWORDS = {
        # Static/slow motion
        "static": 0.0,
        "still": 0.0,
        "stationary": 0.0,
        "motionless": 0.0,
        "slow": 0.2,
        "slowly": 0.2,
        "gentle": 0.2,
        "calm": 0.2,
        # Medium motion
        "moving": 0.5,
        "walk": 0.5,
        "walking": 0.5,
        "moderate": 0.5,
        # Fast motion
        "fast": 0.8,
        "quickly": 0.8,
        "rapid": 0.8,
        "swift": 0.8,
        "run": 0.8,
        "running": 0.8,
        "sprint": 0.9,
        "dash": 0.9,
        "race": 0.9,
        # Very dynamic
        "dynamic": 0.7,
        "energetic": 0.7,
        "action": 0.8,
        "explosive": 0.9,
        "sudden": 0.9,
        # Smooth
        "smooth": 0.4,
        "fluid": 0.4,
        "flowing": 0.4,
    }

    def __init__(self, config=None):
        super().__init__(config)
    def setup(self) -> None:
        pass

    def _extract_expected_motion(self, caption: str) -> float:
        """Extract expected motion level from caption.

        Args:
            caption: Text caption

        Returns:
            Expected motion level (0-1), or 0.5 if no keywords found
        """
        caption_lower = caption.lower()

        # Find all matching keywords
        matched_levels = []
        for keyword, level in self.MOTION_KEYWORDS.items():
            if re.search(r'\b' + keyword + r'\b', caption_lower):
                matched_levels.append(level)

        if not matched_levels:
            # No motion keywords found, assume moderate
            return 0.5

        # Average if multiple keywords
        return float(np.mean(matched_levels))

    def _compute_actual_motion(self, video_path) -> float:
        """Compute actual motion level in video using optical flow.

        Args:
            video_path: Path to video

        Returns:
            Actual motion level (0-1)
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            motion_magnitudes = []
            prev_frame = None
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_frame is not None and frame_count % 3 == 0:  # Sample every 3 frames
                    try:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        motion_magnitudes.append(magnitude.mean())
                    except Exception:
                        pass

                prev_frame = gray
                frame_count += 1

            cap.release()

            if not motion_magnitudes:
                return 0.5

            # Normalize motion magnitude to [0, 1]
            # Empirical max motion magnitude is ~10-15
            avg_motion = np.mean(motion_magnitudes)
            normalized_motion = min(avg_motion / 10.0, 1.0)

            return float(normalized_motion)

        except Exception as e:
            logger.debug(f"Actual motion computation failed: {e}")
            return 0.5

    def process(self, sample: Sample) -> Sample:
        """Process sample to compute dynamics controllability."""
        if not sample.is_video:
            return sample

        # Need caption
        if sample.caption is None or not sample.caption.text:
            return sample

        try:
            caption = sample.caption.text

            # Extract expected motion from caption
            expected_motion = self._extract_expected_motion(caption)

            # Compute actual motion in video
            actual_motion = self._compute_actual_motion(sample.path)

            # Compute controllability as inverse of error
            # If expected ≈ actual, controllability is high
            error = abs(expected_motion - actual_motion)
            controllability = 1.0 - error

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.dynamics_controllability = controllability

            logger.debug(
                f"Dynamics controllability for {sample.path.name}: {controllability:.3f} "
                f"(expected: {expected_motion:.2f}, actual: {actual_motion:.2f})"
            )

        except Exception as e:
            logger.warning(f"Dynamics controllability processing failed for {sample.path}: {e}")

        return sample

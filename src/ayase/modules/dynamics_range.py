"""Dynamics Range module.

Measures the extent of variations in video content (DEVIL protocol).
Assesses how much motion and scene variation occurs in the video.
Range: 0-100 (higher = more dynamic content).
"""

import logging

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DynamicsRangeModule(PipelineModule):
    name = "dynamics_range"
    description = "Measures extent of motion and content variation (DEVIL protocol)"
    default_config = {
        # Pure OpenCV
        "scene_change_threshold": 30.0,  # Threshold for detecting scene changes
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.scene_change_threshold = self.config.get("scene_change_threshold", 30.0)

    def setup(self) -> None:
        pass  # No setup needed

    def process(self, sample: Sample) -> Sample:
        """Process sample to compute dynamics range."""
        if not sample.is_video:
            return sample  # Dynamics range only for videos

        try:
            cap = cv2.VideoCapture(str(sample.path))
            if not cap.isOpened():
                return sample

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Single-pass: read all frames once, compute all three components
            diffs = []
            magnitudes = []
            scene_changes = 0
            prev_gray = None
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    # Component 1: frame difference (every pair)
                    diff = cv2.absdiff(prev_gray, gray).mean()
                    diffs.append(diff)

                    # Component 3: scene change detection (every pair)
                    if diff > self.scene_change_threshold:
                        scene_changes += 1

                    # Component 2: optical flow (every 5th frame)
                    if frame_count % 5 == 0:
                        try:
                            flow = cv2.calcOpticalFlowFarneback(
                                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                            )
                            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                            magnitudes.append(magnitude.mean())
                        except Exception:
                            pass

                prev_gray = gray
                frame_count += 1

            cap.release()

            # Derive per-component results
            diff_variance = float(np.var(diffs)) if diffs else 0.0
            motion_range = float(np.max(magnitudes) - np.min(magnitudes)) if magnitudes else 0.0
            scene_change_freq = scene_changes / max(total_frames, 1) if total_frames > 0 else 0

            # Combine into normalized score
            # Normalize each component (empirical ranges)
            diff_var_norm = min(diff_variance / 100.0, 1.0)
            motion_range_norm = min(motion_range / 10.0, 1.0)
            scene_freq_norm = min(scene_change_freq * 100, 1.0)

            dynamics_range = (diff_var_norm + motion_range_norm + scene_freq_norm) / 3.0 * 100.0

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.dynamics_range = dynamics_range

            logger.debug(
                f"Dynamics range for {sample.path.name}: {dynamics_range:.1f} "
                f"(diff_var: {diff_variance:.1f}, motion: {motion_range:.2f}, scenes: {scene_changes})"
            )

        except Exception as e:
            logger.warning(f"Dynamics range processing failed for {sample.path}: {e}")

        return sample

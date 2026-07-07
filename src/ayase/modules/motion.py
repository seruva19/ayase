"""Motion dynamics analysis via Farneback dense optical flow with effective FPS detection.

Computes mean optical flow magnitude across sampled frame pairs. Also detects
duplicate frames to estimate effective vs container FPS. Returns motion_score."""

import logging
import cv2
import numpy as np
from typing import Optional, List

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class MotionModule(PipelineModule):
    name = "motion"
    description = "Analyzes motion dynamics (optical flow, flickering)"
    default_config = {
        "sample_rate": 5,
        "low_motion_threshold": 0.5,
        "high_motion_threshold": 20.0,
    }
    metric_groups = {
        "motion_score": "motion",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.sample_rate = self.config.get("sample_rate", 5) # Process every Nth frame
        self.low_motion_threshold = self.config.get("low_motion_threshold", 0.5)
        self.high_motion_threshold = self.config.get("high_motion_threshold", 20.0)
        self._backend = "algorithmic"  # Farneback optical flow + duplicate-frame stats

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            self._analyze_motion(sample)
        except Exception as e:
            logger.error(f"Motion analysis failed: {e}")

        return sample

    def _analyze_motion(self, sample: Sample) -> None:
        """Single-pass motion analysis.

        One decode pass computes both the strided Farneback optical-flow motion
        score and, over a contiguous middle window, the duplicate-frame
        ("effective FPS") statistic — so the file is opened only once.
        """
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Effective-FPS window: a contiguous block from the middle used to detect
        # duplicate frames (e.g. upsampled 12fps -> 24fps anime/cartoons).
        fps_window_start = -1
        fps_window_len = 0
        if fps > 0 and total_frames >= 2:
            frames_to_check = int(min(fps, 30))
            if frames_to_check >= 5:
                fps_window_start = total_frames // 2
                fps_window_len = frames_to_check

        prev_sampled_gray = None
        flows = []
        diffs = []

        prev_window_gray = None
        window_unique = 0
        window_read = 0

        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                in_window = (
                    fps_window_len > 0
                    and fps_window_start <= frame_idx < fps_window_start + fps_window_len
                )
                is_sampled = (frame_idx % self.sample_rate == 0)

                gray = None
                if is_sampled or in_window:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if is_sampled:
                    if prev_sampled_gray is not None:
                        # 1. Optical Flow (Farneback) - Dense, between sampled frames
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_sampled_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        flows.append(np.mean(mag))

                        # 2. Pixel Difference (Flickering/Static check)
                        diffs.append(np.mean(cv2.absdiff(prev_sampled_gray, gray)))
                    prev_sampled_gray = gray

                if in_window:
                    # 3. Duplicate-frame detection over consecutive window frames
                    window_read += 1
                    if prev_window_gray is None:
                        window_unique += 1  # First frame is always unique
                    elif np.mean(cv2.absdiff(prev_window_gray, gray)) > 2.0:
                        # 2.0 is a robust middle ground for "new content".
                        window_unique += 1
                    prev_window_gray = gray

                frame_idx += 1
        finally:
            cap.release()

        if not flows:
            return

        avg_motion = float(np.mean(flows))

        # Store motion score in quality metrics
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.motion_score = avg_motion

        # Thresholds (tunable)
        if avg_motion < self.low_motion_threshold:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low motion (static/slideshow): {avg_motion:.2f}",
                    details={"avg_flow": float(avg_motion)},
                    recommendation="Remove static clips or slideshows from the training set as they provide little temporal information."
                )
            )

        if avg_motion > self.high_motion_threshold:
             sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"High motion: {avg_motion:.2f}",
                    details={"avg_flow": float(avg_motion)},
                    recommendation="Check for camera shake or fast movement. Consider stabilizing or discarding if motion blur is excessive."
                )
            )

        # Effective FPS: flag videos whose effective FPS is well below container FPS.
        if fps_window_len > 0 and window_read > 0:
            effective_ratio = window_unique / window_read
            effective_fps = fps * effective_ratio
            if effective_ratio < 0.7:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Low Effective FPS: ~{effective_fps:.1f} (Container: {fps:.1f})",
                        details={
                            "effective_fps": float(effective_fps),
                            "container_fps": float(fps),
                            "fps_ratio": float(effective_ratio),
                        },
                        recommendation="Video contains Duplicate Frames (e.g., upsampled 12fps -> 24fps). Consider downsampling to save compute."
                    )
                )



"""Video/image metadata extraction and validation (resolution, FPS, duration, aspect ratio).

Populates VideoMetadata or ImageMetadata on the sample. Validates against
configurable thresholds for minimum resolution, FPS, and duration bounds."""

import logging
import cv2
from pathlib import Path
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity, VideoMetadata, ImageMetadata
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MetadataModule(PipelineModule):
    name = "metadata"
    description = "Checks video/image metadata (resolution, FPS, duration, integrity)"
    default_config = {
        "min_resolution": 720,
        "min_fps": 15,
        "min_duration": 2.0,
        "max_duration": 60.0,
        "min_aspect_ratio": None,
        "max_aspect_ratio": None,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.min_resolution = self.config.get("min_resolution", 720)
        self.min_fps = self.config.get("min_fps", 15)
        self.min_duration = self.config.get("min_duration", 2.0)
        self.max_duration = self.config.get("max_duration", 60.0)
        self.min_aspect_ratio = self.config.get("min_aspect_ratio")
        self.max_aspect_ratio = self.config.get("max_aspect_ratio")

    def process(self, sample: Sample) -> Sample:
        try:
            if sample.is_video:
                self._process_video(sample)
            else:
                self._process_image(sample)
        except Exception as e:
            logger.error(f"Metadata extraction failed for {sample.path}: {e}")
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Metadata extraction failed: {str(e)}",
                    details={"error": str(e)},
                )
            )
        return sample

    def _process_video(self, sample: Sample) -> None:
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {sample.path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        duration = frame_count / fps if fps > 0 else 0

        sample.video_metadata = VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            file_size=sample.path.stat().st_size,
        )

        # Validation logic
        min_dim = min(width, height)
        if min_dim < self.min_resolution:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low resolution: {width}x{height} (min dimension < {self.min_resolution})",
                    recommendation=f"Upscale content to at least {self.min_resolution}p using AI upscalers (e.g., Real-ESRGAN) or discard.",
                )
            )

        if fps < self.min_fps:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low FPS: {fps:.2f} (min {self.min_fps})",
                    recommendation=f"Interpolate video to increase FPS (e.g., using RIFE) or check source settings.",
                )
            )

        if duration < self.min_duration:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Too short: {duration:.2f}s (min {self.min_duration}s)",
                    recommendation="Discard very short clips as they lack sufficient temporal context for training.",
                )
            )
        elif duration > self.max_duration:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Long video: {duration:.2f}s (max {self.max_duration}s)",
                    recommendation="Split long videos into shorter clips to improve training efficiency.",
                )
            )

        if width > 0 and height > 0:
            aspect_ratio = width / height
            if self.min_aspect_ratio and aspect_ratio < self.min_aspect_ratio:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Narrow aspect ratio: {aspect_ratio:.2f} (min {self.min_aspect_ratio})",
                        details={"aspect_ratio": aspect_ratio},
                        recommendation="Consider resizing or cropping to match the target aspect ratio.",
                    )
                )
            if self.max_aspect_ratio and aspect_ratio > self.max_aspect_ratio:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Wide aspect ratio: {aspect_ratio:.2f} (max {self.max_aspect_ratio})",
                        details={"aspect_ratio": aspect_ratio},
                        recommendation="Consider resizing or cropping to match the target aspect ratio.",
                    )
                )

    def _process_image(self, sample: Sample) -> None:
        img = cv2.imread(str(sample.path))
        if img is None:
            raise IOError(f"Cannot read image file: {sample.path}")

        height, width, channels = img.shape

        sample.image_metadata = ImageMetadata(
            width=width,
            height=height,
            channels=channels,
            format=sample.path.suffix.lstrip(".").lower(),
            file_size=sample.path.stat().st_size,
        )

        min_dim = min(width, height)
        if min_dim < self.min_resolution:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low resolution: {width}x{height} (min dimension < {self.min_resolution})",
                    recommendation=f"Upscale content to at least {self.min_resolution}p using AI upscalers (e.g., Real-ESRGAN) or discard."
                )
            )

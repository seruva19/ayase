"""Overexposure, underexposure, and low contrast detection via luminance histogram analysis.

Checks pixel distribution in the grayscale channel for clipped shadows/highlights
and insufficient dynamic range. No ML dependencies required."""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class ExposureModule(PipelineModule):
    name = "exposure"
    description = "Checks for overexposure, underexposure, and low contrast using histograms"
    default_config = {
        "overexposure_threshold": 0.3,
        "underexposure_threshold": 0.3,
        "contrast_threshold": 30.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        # Configurable thresholds
        self.overexposure_threshold = self.config.get("overexposure_threshold", 0.3) # Max % of pixels > 240
        self.underexposure_threshold = self.config.get("underexposure_threshold", 0.3) # Max % of pixels < 15
        self.contrast_threshold = self.config.get("contrast_threshold", 30.0) # Min std dev of brightness

    def process(self, sample: Sample) -> Sample:
        image = self._load_image(sample)
        if image is None:
            return sample

        try:
            # Convert to Gray/Luma
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            total_pixels = gray.size
            
            # 1. Underexposure Check
            # Count pixels < 15
            dark_pixels = np.count_nonzero(gray < 15)
            dark_ratio = dark_pixels / total_pixels
            
            if dark_ratio > self.underexposure_threshold:
                 sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Severe Underexposure: {dark_ratio*100:.1f}% pixels are dark",
                        details={"dark_ratio": float(dark_ratio)},
                        recommendation="Adjust brightness levels or apply gamma correction. Discard if details are lost in shadows."
                    )
                )

            # 2. Overexposure Check
            # Count pixels > 240
            bright_pixels = np.count_nonzero(gray > 240)
            bright_ratio = bright_pixels / total_pixels
            
            if bright_ratio > self.overexposure_threshold:
                 sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Severe Overexposure: {bright_ratio*100:.1f}% pixels are bright (blown out)",
                        details={"bright_ratio": float(bright_ratio)},
                        recommendation="Reduce highlights or apply tone mapping. Discard if highlights are clipped and unrecoverable."
                    )
                )

            # 3. Low Contrast Check
            # Standard deviation of the histogram
            std_dev = np.std(gray)
            if std_dev < self.contrast_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Low Contrast: {std_dev:.1f} (Image might look washed out or flat)",
                        details={"contrast_std": float(std_dev)},
                        recommendation="Increase contrast or use histogram equalization to improve dynamic range."
                    )
                )

        except Exception as e:
            logger.warning(f"Exposure check failed: {e}")

        return sample

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
                ret, frame = cap.read()
                cap.release()
                return frame if ret else None
            else:
                return cv2.imread(str(sample.path))
        except Exception:
            return None

"""Temporal style consistency via HSV histogram correlation across frames.

Lightweight proxy for Gram Matrix consistency. Compares color distributions
between sampled frames. Values below 0.8 indicate style drift."""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class StyleConsistencyModule(PipelineModule):
    name = "style_consistency"
    description = "Appearance Style verification (Gram Matrix Consistency)"
    default_config = {}

    def __init__(self, config=None):
        super().__init__(config)

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            # We use Gram Matrices of deep features (VGG) usually for style.
            # But that's heavy.
            # Lightweight proxy: Color Histogram Correlation over time.
            # If color distribution changes drastically, style is inconsistent.
            
            consistency_score = self._analyze_histogram_consistency(sample)
            
            if consistency_score < 0.8:
                 sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low Style/Color Consistency: {consistency_score:.2f}",
                        details={"style_consistency": float(consistency_score)}
                    )
                )

        except Exception as e:
            logger.warning(f"Style consistency check failed: {e}")

        return sample

    def _analyze_histogram_consistency(self, sample: Sample) -> float:
        max_frames = self.config.get("max_frames", 300)
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return 1.0

        prev_hist = None
        correlations = []
        sampled = 0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 10 != 0:
                frame_idx += 1
                continue
            frame_idx += 1
            sampled += 1

            if sampled > max_frames:
                break

            # Calculate HSV histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            if prev_hist is not None:
                score = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                correlations.append(score)

            prev_hist = hist

        cap.release()
        
        if not correlations:
            return 1.0
            
        return np.mean(correlations)

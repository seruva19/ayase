"""Cumulative Probability of Blur Detection (CPBD) for perceptual sharpness assessment.

Uses the ``cpbd`` library (Narvekar & Karam's published CPBD metric). When the
library is not installed the metric is left unset (no heuristic fallback).
Returns cpbd_score (0-1, higher = sharper). Warns on the blurriest frame.
"""

import logging
import numpy as np

from ayase.image import sample_frames
from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CPBDModule(PipelineModule):
    name = "cpbd"
    description = "Cumulative Probability of Blur Detection (Perceptual Blur)"
    default_config = {
        "threshold_cpbd": 0.65,
        "max_frames": 8,
    }
    metric_groups = {
        "cpbd_score": "basic",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.threshold_cpbd = self.config.get("threshold_cpbd", 0.65)
        self.max_frames = self.config.get("max_frames", 8)
        self._cpbd_available = False
        self.cpbd = None
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import cpbd

            self.cpbd = cpbd
            self._cpbd_available = True
            self._backend = "cpbd"
            logger.info("CPBD initialised")
        except ImportError:
            logger.warning("cpbd not installed; CPBD metric skipped. Install with: pip install cpbd")

    def process(self, sample: Sample) -> Sample:
        if not self._cpbd_available:
            return sample

        frames = sample_frames(sample.path, max_frames=self.max_frames, color="gray")
        if not frames:
            return sample

        scores = []
        for gray in frames:
            try:
                scores.append(float(self.cpbd.compute(np.ascontiguousarray(gray))))
            except Exception as e:
                logger.warning(f"CPBD check failed: {e}")
                continue

        if not scores:
            return sample

        # CPBD: Higher is better (Sharpness). We report the AVERAGE and
        # flag the WORST frame (blurriest).
        min_score = min(scores)
        avg_score = sum(scores) / len(scores)

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.cpbd_score = float(avg_score)

        if min_score < self.threshold_cpbd:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"High perceptual blur detected (Min CPBD: {min_score:.2f})",
                    details={"cpbd_min": min_score, "cpbd_avg": avg_score},
                    recommendation="Discard blurry image or attempt sharpening.",
                )
            )

        return sample

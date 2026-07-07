"""Usability Rate module.

Post-processing module that computes the percentage of usable frames/samples
based on quality thresholds. Compares predictions vs MOS if available.
Range: 0-100 (percentage of usable content).
"""

import logging
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class UsabilityRateModule(PipelineModule):
    name = "usability_rate"
    description = "Computes percentage of usable frames based on quality thresholds"
    default_config = {
        "quality_threshold": 50.0,  # Minimum quality score to be "usable"
    }
    metric_groups = {
        "usability_rate": "meta",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.quality_threshold = self.config.get("quality_threshold", 50.0)

    def setup(self) -> None:
        pass

    def _compute_frame_usability(self, sample: Sample) -> Optional[float]:
        """Compute usability based on available quality metrics.

        For videos, this would need frame-level metrics (not implemented here).
        For samples, we check if overall quality meets threshold. Returns None
        when no usable quality signal is available rather than a fabricated
        neutral value.
        """
        if sample.quality_metrics is None:
            return None  # Unknown — no quality signal

        # Check technical score
        if sample.quality_metrics.technical_score is not None:
            return 1.0 if sample.quality_metrics.technical_score >= self.quality_threshold else 0.0

        # Check aesthetic score as fallback
        if sample.quality_metrics.aesthetic_score is not None:
            # Aesthetic score is 0-10, convert to 0-100
            aesthetic_100 = sample.quality_metrics.aesthetic_score * 10
            return 1.0 if aesthetic_100 >= self.quality_threshold else 0.0

        return None  # Unknown — no quality signal

    def process(self, sample: Sample) -> Sample:
        """Process sample to compute usability rate."""
        try:
            usability = self._compute_frame_usability(sample)
            if usability is None:
                # No usable quality signal — leave usability_rate unset
                # rather than reporting a fabricated 50%.
                return sample

            # Store as percentage
            usability_rate = usability * 100.0

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.usability_rate = usability_rate

            logger.debug(f"Usability rate for {sample.path.name}: {usability_rate:.0f}%")

        except Exception as e:
            logger.warning(f"Usability rate processing failed for {sample.path}: {e}")

        return sample

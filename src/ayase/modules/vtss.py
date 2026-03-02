"""VTSS (Video Training Suitability Score) module.

Inspired by Koala-36M (CVPR 2025). Meta-metric that combines
multiple sub-metrics to predict overall training data quality.
Aggregates existing quality signals into a single suitability score.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VTSSModule(PipelineModule):
    name = "vtss"
    description = "Video Training Suitability Score (0-1, meta-metric)"
    default_config = {
        "weights": {
            "aesthetic": 0.15,
            "technical": 0.15,
            "motion": 0.10,
            "temporal_consistency": 0.15,
            "blur": 0.10,
            "noise": 0.10,
            "scene_stability": 0.10,
            "resolution": 0.15,
        }
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            qm = sample.quality_metrics
            weights = self.config.get("weights", self.default_config["weights"])
            scores = []
            total_weight = 0.0

            # Aesthetic quality (normalize various aesthetic scores to 0-1)
            aes = self._get_aesthetic(qm)
            if aes is not None:
                scores.append(weights.get("aesthetic", 0.15) * aes)
                total_weight += weights.get("aesthetic", 0.15)

            # Technical quality
            tech = self._get_technical(qm)
            if tech is not None:
                scores.append(weights.get("technical", 0.15) * tech)
                total_weight += weights.get("technical", 0.15)

            # Motion quality (not too static, not too chaotic)
            motion = self._get_motion_quality(qm)
            if motion is not None:
                scores.append(weights.get("motion", 0.10) * motion)
                total_weight += weights.get("motion", 0.10)

            # Temporal consistency
            tc = qm.temporal_consistency
            if tc is not None:
                scores.append(weights.get("temporal_consistency", 0.15) * min(1.0, tc))
                total_weight += weights.get("temporal_consistency", 0.15)

            # Sharpness (inverse of blur)
            if qm.blur_score is not None:
                sharpness = min(1.0, qm.blur_score / 500.0)
                scores.append(weights.get("blur", 0.10) * sharpness)
                total_weight += weights.get("blur", 0.10)

            # Low noise
            if qm.noise_score is not None:
                low_noise = max(0.0, 1.0 - qm.noise_score / 50.0)
                scores.append(weights.get("noise", 0.10) * low_noise)
                total_weight += weights.get("noise", 0.10)

            # Scene stability
            if qm.scene_stability is not None:
                scores.append(weights.get("scene_stability", 0.10) * qm.scene_stability)
                total_weight += weights.get("scene_stability", 0.10)

            if total_weight > 0:
                sample.quality_metrics.vtss = float(sum(scores) / total_weight)
        except Exception as e:
            logger.warning("VTSS computation failed: %s", e)
        return sample

    def _get_aesthetic(self, qm) -> Optional[float]:
        """Get best available aesthetic score, normalized to 0-1."""
        if qm.laion_aesthetic is not None:
            return min(1.0, qm.laion_aesthetic / 10.0)
        if qm.aesthetic_score is not None:
            return min(1.0, qm.aesthetic_score / 10.0)
        if qm.dover_aesthetic is not None:
            return min(1.0, max(0.0, qm.dover_aesthetic))
        return None

    def _get_technical(self, qm) -> Optional[float]:
        """Get best available technical quality, normalized to 0-1."""
        if qm.dover_technical is not None:
            return min(1.0, max(0.0, qm.dover_technical))
        if qm.technical_score is not None:
            return min(1.0, qm.technical_score / 100.0)
        return None

    def _get_motion_quality(self, qm) -> Optional[float]:
        """Motion quality: prefer moderate motion, penalise static or chaotic."""
        ms = qm.motion_score
        if ms is None:
            return None
        # Bell curve: peak at moderate motion (~5-15), low for static or extreme
        return float(np.exp(-0.5 * ((ms - 10.0) / 8.0) ** 2))

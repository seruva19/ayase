"""SAMA — Scaling and Masking for Video Quality Assessment (2024).

Patch pyramid with a masking strategy for local + global quality, improving on
FAST-VQA.

Only the real trained SAMA model produces ``sama_score``. Earlier revisions ran
a ResNet-50 backbone through randomly-initialised attention and regression heads
— those heads are untrained, so their output is meaningless and has been
removed. No trained SAMA weights are wired up here, so the score is left
``None``.

sama_score — higher = better quality
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SAMAModule(PipelineModule):
    name = "sama"
    description = "SAMA scaling+masking VQA (real model only)"
    default_config = {
        "subsample": 8,
        "mask_ratio": 0.5,
    }
    metric_groups = {
        "sama_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.mask_ratio = self.config.get("mask_ratio", 0.5)
        self._backend = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning("SAMA: real trained model unavailable; sama_score left unset.")

    def process(self, sample: Sample) -> Sample:
        # No trained SAMA weights available; do not fabricate a score.
        return sample

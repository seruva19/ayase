"""SiamVQA — Siamese Network for High-Resolution VQA (arXiv 2025).

Siamese network sharing weights between aesthetic and technical branches.

Only the real trained SiamVQA model produces ``siamvqa_score``. Earlier
revisions ran a ResNet-50 backbone through randomly-initialised technical,
aesthetic, fusion and pooling heads — those heads are untrained, so their output
is meaningless and has been removed. No trained SiamVQA weights are wired up
here, so the score is left ``None``.

Paper: https://arxiv.org/html/2503.02330

siamvqa_score — higher = better quality
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SiamVQAModule(PipelineModule):
    name = "siamvqa"
    description = "SiamVQA Siamese high-resolution VQA (real model only)"
    default_config = {
        "subsample": 8,
        "num_crops": 5,
        "crop_size": 224,
    }
    metric_groups = {
        "siamvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.num_crops = self.config.get("num_crops", 5)
        self.crop_size = self.config.get("crop_size", 224)
        self._backend = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning("SiamVQA: real trained model unavailable; siamvqa_score left unset.")

    def process(self, sample: Sample) -> Sample:
        # No trained SiamVQA weights available; do not fabricate a score.
        return sample

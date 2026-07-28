"""SiamVQA — Siamese Network for High-Resolution VQA (arXiv 2025).

Siamese network sharing weights between aesthetic and technical branches.

Only the real trained SiamVQA model produces ``siamvqa_score``. Randomly
initialised technical, aesthetic, fusion, and pooling heads do not produce a
meaningful SiamVQA score. No public trained SiamVQA weights are available, so
the score is left ``None``.

Paper: https://arxiv.org/html/2503.02330

siamvqa_score — higher = better quality

Backend requirements
Metric: SiamVQA (ICASSP 2025).
Unavailable because: Samsung paper, no public release; the Simple-Siamese high-res model must be trained.
Source: SiamVQA, ICASSP 2025 (no public release).
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SiamVQAModule(PipelineModule):
    name = "siamvqa"
    requires_external_backend = True  # no turnkey real backend in a standard install
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

"""SR4KVQA — Super-Resolution 4K Video Quality (2024).

SR4KVQA targets quality assessment of super-resolved 4K content, predicting a
learned quality score calibrated against subjective ratings of SR artifacts
(ringing, texture loss, aliasing, hallucination).

SR4KVQA requires trained artifact and quality heads. No public trained
SR4KVQA checkpoint is bundled, so the module reports the metric as unavailable
and leaves ``sr4kvqa_score`` unset rather than emitting an untrained score.

Backend requirements
Metric: SR4KVQA (2024).
Unavailable because: Repo links only a Swin-T *init*, no trained ckpt.
Source: SR4KVQA, 2024 (Swin-T init only, no trained ckpt).
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SR4KVQAModule(PipelineModule):
    name = "sr4kvqa"
    provenance = "utility"
    requires_external_backend = True  # no turnkey real backend in a standard install
    description = "SR4KVQA super-resolution 4K quality (2024)"
    default_config = {
        "subsample": 8,
        "patch_size": 224,
        "max_patches": 9,
    }
    metric_groups = {
        "sr4kvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        self._ml_available = False
        self._backend = "unavailable"
        logger.info(
            "SR4KVQA: no publicly available trained checkpoint; the learned "
            "quality head cannot be reproduced. Metric reported as unavailable."
        )

    def process(self, sample: Sample) -> Sample:
        # No trained SR4KVQA head available -> leave sr4kvqa_score unset instead
        # of fabricating a value from randomly-initialised layers.
        return sample

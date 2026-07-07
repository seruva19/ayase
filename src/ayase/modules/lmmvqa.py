"""LMM-VQA -- Large Multimodal Model VQA.

LMM-VQA is a published spatiotemporal VQA model built on large-multimodal-model
features. Ayase does not ship a loadable LMM-VQA checkpoint, so this module has
no genuine backend.

Per the project's no-heuristic policy, ``lmmvqa_score`` must come from the real
model or be left ``None``. It must never be fabricated from CLIP quality-prompt
similarity or from an ImageNet backbone with randomly-initialised, untrained
regression/attention heads. Until a real LMM-VQA backend is wired in, the score
is left unset.

lmmvqa_score -- higher = better quality (0-1); real model only.
"""

import logging
from typing import Optional

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class LMMVQAModule(PipelineModule):
    name = "lmmvqa"
    provisional = True  # no turnkey real backend in a standard install
    description = "LMM-VQA spatiotemporal quality (real model only; disabled if unavailable)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "lmmvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._device = "cpu"
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        # No public, loadable LMM-VQA checkpoint is wired in. Rather than
        # fabricate a score from a CLIP proxy or untrained head, disable it.
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "LMM-VQA: no real LMM-VQA model/weights available; module disabled "
            "(lmmvqa_score left unset). Wire a genuine LMM-VQA checkpoint to enable."
        )

    def process(self, sample: Sample) -> Sample:
        # Real backend unavailable -> leave lmmvqa_score None (graceful).
        return sample

    def _compute_quality(self, sample: Sample) -> Optional[float]:
        return None

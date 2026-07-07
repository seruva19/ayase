"""InternVQA -- Lightweight Compressed Video Quality Assessment.

InternVQA is a published no-reference VQA model for compressed video. Ayase
does not ship (and cannot currently locate a loadable checkpoint for) the real
InternVQA weights, so this module has no genuine backend.

Per the project's no-heuristic policy, a named-metric field must be produced by
the real model or left ``None`` -- it must never be fabricated from an
ImageNet backbone with randomly-initialised, untrained quality heads. Until a
real InternVQA backend is wired in, ``internvqa_score`` is left unset.

internvqa_score -- higher = better quality (0-1); real model only.
"""

import logging
from typing import Optional

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class InternVQAModule(PipelineModule):
    name = "internvqa"
    provisional = True  # no turnkey real backend in a standard install
    description = "InternVQA compressed-video quality (real model only; disabled if unavailable)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "internvqa_score": "nr_quality",
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
        # No public, loadable InternVQA checkpoint is wired in. Rather than
        # fabricate a score from an untrained head, disable the module.
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "InternVQA: no real InternVQA model/weights available; module disabled "
            "(internvqa_score left unset). Wire a genuine InternVQA checkpoint to enable."
        )

    def process(self, sample: Sample) -> Sample:
        # Real backend unavailable -> leave internvqa_score None (graceful).
        return sample

    def _compute_quality(self, sample: Sample) -> Optional[float]:
        return None

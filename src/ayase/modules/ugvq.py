"""UGVQ — Unified Generated Video Quality (ACM TOMM 2024).

Multi-dimensional quality assessment for generated video content.

The published UGVQ model is trained on the LGVQ dataset with a dedicated
multi-branch architecture. There is no installable UGVQ backend, and a
CLIP zero-shot prompt comparison ("high quality photo" vs "blurry photo")
is a proxy — not UGVQ — so it is not emitted under the UGVQ name. This
module reports itself unavailable until a real UGVQ backend is wired in.

Output field: ``ugvq_score`` (populated only with a real backend).

REVIVAL NOTES (provisional — no turnkey backend)
Metric: UGVQ / LGVQ (ACM TOMM 2024).
Category: TRAINING-ONLY.
Why provisional: No shipped weights, but dataset AND reference code are on GitHub — mostly wiring + a run.
To revive: Use the released reference code + LGVQ dataset (2,808 AIGC videos, public) to train the
  3-dim (spatial/temporal/text-align) model; validate you reproduce the paper's SRCC/PLCC before
  flipping provisional=False. Effort M; genuine AIGC gap-filler.
Source: UGVQ / LGVQ, ACM TOMM 2024 (dataset + reference code on GitHub).
"""

import logging
from typing import Optional

from ayase.models import Sample, QualityMetrics  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class UGVQModule(PipelineModule):
    name = "ugvq"
    provisional = True  # no turnkey real backend in a standard install
    description = "UGVQ unified generated video quality (TOMM 2024)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }
    metric_groups = {
        "ugvq_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "UGVQ unavailable: no trained UGVQ (LGVQ) model is installable; "
            "ugvq_score will not be populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        return None

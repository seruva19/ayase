"""UNQA — Unified No-Reference Quality Assessment (2024).

A unified framework that shares feature representations across audio,
image, and video modalities for quality prediction.

The published UNQA method relies on a *trained* unified quality head. No
official UNQA weights are distributed as an installable package, and a
ResNet backbone with a randomly-initialised (untrained) regression head
does not produce a meaningful UNQA score. Rather than emit a fabricated
number, this module reports itself unavailable until a real UNQA backend
is wired in.

Output field: ``confidence_score`` (populated only with a real backend).

REVIVAL NOTES (provisional — no turnkey backend)
Metric: UNQA (2024).
Category: TRAINING-ONLY.
Why provisional: No installable weights; the unified rank-based joint head must be trained.
To revive: Reimplement the unified audio/image/video/AV NR-QA rank-based joint training; train on the
  joint public A/I/V/AV QA DBs; validate you reproduce the paper's SRCC/PLCC before flipping
  provisional=False. Redundant with ayase's per-modality metrics.
Source: UNQA, 2024 (no released weights).
"""

import logging
from typing import Optional

from ayase.models import Sample, QualityMetrics  # noqa: F401 (QualityMetrics kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class UNQAModule(PipelineModule):
    name = "unqa"
    provisional = True  # no turnkey real backend in a standard install
    description = "UNQA unified no-reference quality for audio/image/video (2024)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "confidence_score": "meta",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        # No installable UNQA model exists; a random-init head would fabricate
        # scores, so the module stays unavailable rather than emit noise.
        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "UNQA unavailable: no trained unified quality model is bundled; "
            "confidence_score will not be populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        return sample

    def _compute_quality(self, sample: Sample) -> Optional[float]:
        return None

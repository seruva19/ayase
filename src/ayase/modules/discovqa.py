"""DisCoVQA — Temporal Distortion-Content Transformers for VQA.

IEEE 2023 — separates temporal distortion extraction from content-aware
temporal attention using transformers.

DisCoVQA is a *trained* transformer model (content/distortion dual-path
networks plus a learned quality head). There is no maintained Python package or
portable set of weights for it, and a randomly-initialised reimplementation
would produce meaningless scores. The metric is therefore left unavailable
until the upstream DisCoVQA weights are wired in.

GitHub: https://github.com/VQAssessment/DisCoVQA

discovqa_score — higher = better quality (0-1); left None when unavailable.

REVIVAL NOTES (requires_external_backend — no turnkey backend)
Metric: DisCoVQA (2023).
Category: TRAINING-ONLY.
Why requires_external_backend: Repo is an explicit "code released later" stub; no weights.
To revive: Reimplement the distortion-content Swin-T transformer; train on LSVQ + standard VQA
  benchmarks; validate you reproduce the paper's SRCC/PLCC before flipping requires_external_backend=False. Effort M.
Source: https://github.com/VQAssessment/DisCoVQA
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DisCoVQAModule(PipelineModule):
    name = "discovqa"
    requires_external_backend = True  # no turnkey real backend in a standard install
    description = "DisCoVQA temporal distortion-content VQA (2023)"
    default_config = {
        "subsample": 8,
        "frame_size": 224,
    }
    metric_groups = {
        "discovqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.frame_size = self.config.get("frame_size", 224)
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return
        # No maintained package ships the trained DisCoVQA transformer/weights.
        # Refuse to fabricate a score from an untrained proxy network.
        logger.warning(
            "DisCoVQA unavailable: no maintained package provides the trained "
            "DisCoVQA weights (github.com/VQAssessment/DisCoVQA). "
            "discovqa_score left unset."
        )

    def process(self, sample: Sample) -> Sample:
        return sample

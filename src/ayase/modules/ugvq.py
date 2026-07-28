"""UGVQ — Unified Generated Video Quality (ACM TOMM 2024).

Multi-dimensional quality assessment for generated video content.

The published UGVQ model is trained on the LGVQ dataset with a dedicated
multi-branch architecture. As of July 2026, the upstream repository publishes
only the LGVQ dataset link: it contains no model code, weights, or releases.
There is therefore no reproducible upstream UGVQ backend, and a
CLIP zero-shot prompt comparison ("high quality photo" vs "blurry photo")
is a proxy — not UGVQ — so it is not emitted under the UGVQ name. This
module reports itself unavailable until a real UGVQ backend is wired in.

Output field: ``ugvq_score`` (populated only with a real backend).

REVIVAL NOTES (requires_external_backend — upstream backend not released)
Metric: UGVQ / LGVQ (ACM TOMM 2024).
Category: TRAINING-ONLY.
Why requires_external_backend: The dataset is public, but the claimed model/code release is absent.
To revive: Wait for upstream inference code and weights, then reproduce the paper's
  spatial/temporal/alignment SRCC and PLCC before flipping requires_external_backend=False.
Source: UGVQ / LGVQ, ACM TOMM 2024; upstream GitHub repository audited 2026-07-27.
"""

import logging
from typing import Optional

from ayase.models import Sample, QualityMetrics  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class UGVQModule(PipelineModule):
    name = "ugvq"
    requires_external_backend = True  # upstream repository has no model code or weights
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
            "UGVQ unavailable: upstream repository has no inference code or weights; "
            "ugvq_score will not be populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        return None

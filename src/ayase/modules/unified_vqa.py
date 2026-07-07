"""Unified-VQA -- Unified Video Quality Assessment (FR+NR Multi-task).

The "Unified-VQA" score presented here was derived from statistics of CLIP
(or ResNet) frame embeddings — feature-norm magnitude, cross-frame
consistency and cosine smoothness — combined with hand-tuned normalisation
constants. That is an invented, uncalibrated statistic, not a trained
quality model, so it is not emitted under a VQA-metric name.

There is no installable Unified-VQA backend to fall back on, so this module
reports itself unavailable. It also no longer writes ``dover_score`` — that
field belongs solely to the DOVER module.

Output field: ``unified_vqa_score`` (populated only with a real backend).
"""

import logging
from typing import List, Optional

import numpy as np  # noqa: F401 (kept for API parity)

from ayase.models import QualityMetrics, Sample  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class UnifiedVQAModule(PipelineModule):
    name = "unified_vqa"
    provisional = True  # no turnkey real backend in a standard install
    description = "Unified-VQA FR+NR multi-task quality assessment (2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "ViT-B/32",
    }
    metric_info = {
        "unified_vqa_score": "Unified-VQA FR/NR quality score (0-1, higher=better)",
    }
    metric_groups = {
        "unified_vqa_score": "nr_quality",
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
            "Unified-VQA unavailable: no trained Unified-VQA model is bundled and "
            "CLIP/ResNet feature statistics are not a calibrated quality score; "
            "unified_vqa_score will not be populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        return sample

    def _compute_quality(
        self,
        dist_features: List[np.ndarray],
        ref_features: Optional[List[np.ndarray]] = None,
    ) -> Optional[float]:
        return None

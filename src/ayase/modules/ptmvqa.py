"""PTM-VQA — Pre-Trained Model fusion VQA.

CVPR 2024 — integrates features from multiple frozen pre-trained models with
ICID loss, then maps them to quality through a *trained* regression head.

The backbones (CLIP/DINOv2/ResNet-50) are pretrained, but PTM-VQA's fusion /
quality head is trained and no public weights are wired into Ayase. Emitting a
score from an untrained head would be fabrication, so this module reports no
score. Wire trained PTM-VQA weights to enable it.

Paper: https://arxiv.org/abs/2405.17765

ptmvqa_score — higher = better quality (0-1)
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PTMVQAModule(PipelineModule):
    name = "ptmvqa"
    provisional = True  # no turnkey real backend in a standard install
    description = "PTM-VQA multi-PTM fusion VQA (CVPR 2024)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }
    metric_groups = {
        "ptmvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = "unavailable"
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        self._backend = "unavailable"
        logger.info(
            "PTM-VQA unavailable: no pretrained PTM-VQA fusion/quality head wired; "
            "no score emitted."
        )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        # Kept for field discovery; only reached when a real backend is wired.
        score = self._compute_score(sample)
        if score is not None:
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.ptmvqa_score = float(np.clip(score, 0.0, 1.0))

        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        # No trained PTM-VQA head is available; never fabricate a score.
        return None

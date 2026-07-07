"""ProVQA -- Progressive Blind 360 VQA (2022).

Progressive blind video quality assessment for 360-degree content, operating
at multiple resolution levels (pixel -> patch -> frame -> video) with a
*trained* multi-level quality network.

No public pretrained ProVQA weights are wired into Ayase, so this module does
not emit a score: the progressive quality heads would otherwise be untrained
(randomly initialised) and their output fabricated. Wire trained ProVQA
weights to enable it.

provqa_score -- higher = better quality (0-1)
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ProVQAModule(PipelineModule):
    name = "provqa"
    description = "ProVQA progressive blind 360 VQA (2022)"
    default_config = {
        "subsample": 8,
        "n_fine_crops": 6,
    }
    metric_groups = {
        "provqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.n_fine_crops = self.config.get("n_fine_crops", 6)
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return
        self._backend = "unavailable"
        logger.info(
            "ProVQA unavailable: no pretrained ProVQA weights wired; no score emitted."
        )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        # Kept for field discovery; only reached when a real backend is wired.
        score = self._compute_progressive_quality(sample)
        if score is not None:
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.provqa_score = float(np.clip(score, 0.0, 1.0))

        return sample

    def _compute_progressive_quality(self, sample: Sample) -> Optional[float]:
        # No real ProVQA backend is available; never fabricate a score.
        return None

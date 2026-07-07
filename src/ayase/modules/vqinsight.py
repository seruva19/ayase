"""VQ-Insight — ByteDance AIGC Video Quality (AAAI 2026).

Multi-dimensional AIGC video quality assessment model. This module requires
the released VQ-Insight model; when that backend is not installed the metric
is left ``None`` — no CLIP zero-shot proxy or handcrafted approximation is
substituted for the published metric.

vqinsight_score — higher = better (0-1)
"""

import logging
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VQInsightModule(PipelineModule):
    name = "vqinsight"
    description = "VQ-Insight ByteDance multi-dim AIGC scoring (AAAI 2026)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "vqinsight_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._ml_available = False
        self._backend = None
        self._model = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import vqinsight  # type: ignore  # official VQ-Insight backend

            self._model = vqinsight
            self._ml_available = True
            self._backend = "vqinsight"
            logger.info("VQ-Insight initialised (vqinsight backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "VQ-Insight unavailable: the released VQ-Insight model is not installed"
        )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "vqinsight":
            return sample

        try:
            predict = getattr(self._model, "predict", None)
            if predict is None:
                return sample
            score = predict(str(sample.path))
            if score is not None:
                sample.quality_metrics.vqinsight_score = float(score)
        except Exception as e:
            logger.warning("VQ-Insight failed for %s: %s", sample.path, e)
        return sample

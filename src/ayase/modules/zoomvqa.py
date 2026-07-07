"""Zoom-VQA -- Patches, Frames and Clips Integration for VQA.

Zhang et al. CVPRW 2023 -- dual-branch late-fusion architecture (cpnet_multi
IQA branch + Video Swin Transformer VQA branch) for blind video quality
assessment.

GitHub: https://github.com/k-zha14/Zoom-VQA

This module requires the released Zoom-VQA model. When it is not installed the
metric is left ``None``: a generic ImageNet ResNet-50 with untrained,
randomly-initialised quality heads and a temporal-conv stand-in is NOT the
published Zoom-VQA network and would not reproduce it, so no proxy score is
substituted.

zoomvqa_score -- higher = better quality (0-1)
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ZoomVQAModule(PipelineModule):
    name = "zoomvqa"
    description = (
        "Zoom-VQA dual-branch IQA+VQA late-fusion blind VQA (CVPRW 2023)"
    )
    default_config = {
        "subsample": 16,
    }
    metric_groups = {
        "zoomvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 16)
        self._ml_available = False
        self._backend = None
        self._model = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import zoomvqa  # type: ignore  # official Zoom-VQA backend

            self._model = zoomvqa
            self._ml_available = True
            self._backend = "zoomvqa"
            logger.info("Zoom-VQA initialised (zoomvqa backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "Zoom-VQA unavailable: the released Zoom-VQA model is not installed"
        )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "zoomvqa":
            return sample

        try:
            predict = getattr(self._model, "predict", None)
            if predict is None:
                return sample
            score = predict(str(sample.path))
            if score is not None:
                sample.quality_metrics.zoomvqa_score = float(score)
        except Exception as e:
            logger.warning("Zoom-VQA failed for %s: %s", sample.path, e)
        return sample

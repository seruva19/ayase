"""Ada-DQA -- Adaptive Diverse Quality-aware Feature Acquisition (ACM MM 2023).

Ada-DQA learns adaptive quality-aware feature acquisition from a diverse set of
pre-trained models and a trained quality regression head. A generic ImageNet
ResNet-50 backbone paired with an untrained quality head is not Ada-DQA — the
untrained head produces meaningless weights — so no score is emitted under the
Ada-DQA name. This module reports itself unavailable until a real Ada-DQA
backend (with trained weights) is wired in.

Output field: ``adadqa_score`` (populated only with a real backend).
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AdaDQAModule(PipelineModule):
    name = "adadqa"
    description = "Ada-DQA adaptive diverse quality feature VQA (ACM MM 2023)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "adadqa_score": "nr_quality",
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
            import adadqa  # type: ignore  # official Ada-DQA backend (trained weights)

            self._model = adadqa
            self._ml_available = True
            self._backend = "adadqa"
            logger.info("Ada-DQA initialised (adadqa backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "Ada-DQA unavailable: no trained Ada-DQA model is installable; "
            "adadqa_score will not be populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "adadqa":
            return sample

        try:
            predict = getattr(self._model, "predict", None)
            if predict is None:
                return sample
            score = predict(str(sample.path))
            if score is not None:
                sample.quality_metrics.adadqa_score = float(score)
        except Exception as e:
            logger.warning("Ada-DQA failed for %s: %s", sample.path, e)
        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        return None

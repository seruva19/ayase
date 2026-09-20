"""No-reference video quality prediction through an optional Ada-DQA backend.

The published method acquires quality-aware features from diverse frozen models
and distils them into a lighter VQA model. Ayase ships neither trained weights
nor a proxy: it delegates the whole video path to an external ``adadqa.predict``
function and otherwise leaves ``adadqa_score`` unset. A returned score is passed
through unchanged, so its direction and range are properties of that backend.

Basis: https://arxiv.org/abs/2308.00729
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AdaDQAModule(PipelineModule):
    name = "adadqa"
    requires_external_backend = True  # no turnkey real backend in a standard install
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
            import adadqa  # type: ignore  # upstream Ada-DQA backend (trained weights)

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

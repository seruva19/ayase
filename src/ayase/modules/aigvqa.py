"""AIGVQA --- Multi-Dimensional AI-Generated VQA (ICCVW 2025).

GitHub: https://github.com/IntMeGroup/AIGVQA

The published AIGVQA model is a trained multi-dimensional quality network. There
is no installable AIGVQA backend, and a CLIP multi-prompt proxy (spatial /
temporal / aesthetic zero-shot contrasts) is not AIGVQA, so it is not emitted
under the AIGVQA name. This module reports itself unavailable until a real
AIGVQA backend is wired in.

Output field: ``aigvqa_score`` (populated only with a real backend).
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AIGVQAModule(PipelineModule):
    name = "aigvqa"
    description = "AIGVQA multi-dimensional AIGC VQA (ICCVW 2025)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "aigvqa_score": "nr_quality",
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
            import aigvqa  # type: ignore  # official AIGVQA backend

            self._model = aigvqa
            self._ml_available = True
            self._backend = "aigvqa"
            logger.info("AIGVQA initialised (aigvqa backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "AIGVQA unavailable: no trained AIGVQA model is installable; "
            "aigvqa_score will not be populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "aigvqa":
            return sample

        try:
            predict = getattr(self._model, "predict", None)
            if predict is None:
                return sample
            score = predict(str(sample.path))
            if score is not None:
                sample.quality_metrics.aigvqa_score = float(score)
        except Exception as e:
            logger.warning("AIGVQA failed for %s: %s", sample.path, e)
        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        return None

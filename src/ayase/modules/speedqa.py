"""SpEED-QA — Spatial Efficient Entropic Differencing for Quality Assessment.

Bampis et al. 2017 — a reduced-reference quality predictor based on the spatial
entropic differences between a reference and a distorted signal.

Only the real SpEED-QA index produces ``speedqa_score``. Earlier revisions
computed a no-reference score from consecutive-frame local-entropy differences
combined with hand-tuned weights — that is not the published (reduced-reference)
SpEED-QA and has been removed. No real SpEED-QA backend is wired up here, so the
score is left ``None``.

speedqa_score — higher = better quality
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SpEEDQAModule(PipelineModule):
    name = "speedqa"
    description = "SpEED-QA spatial efficient entropic differencing (real model only)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "speedqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = None
        self._model = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import speedqa

            self._model = speedqa
            self._backend = "speedqa_pkg"
            logger.info("SpEED-QA (speedqa package) initialised")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        logger.warning("SpEED-QA: real reduced-reference backend unavailable; speedqa_score left unset.")

    def process(self, sample: Sample) -> Sample:
        if self._backend != "speedqa_pkg" or self._model is None:
            return sample

        try:
            from ayase.models import QualityMetrics

            score = self._model.predict(str(sample.path))
            if score is None:
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.speedqa_score = float(score)
        except Exception as e:
            logger.warning("SpEED-QA failed for %s: %s", sample.path, e)

        return sample

"""CRAVE --- Content-Rich AIGC Video Evaluator (2025).

GitHub: https://github.com/littlespray/CRAVE

Designed for Sora-era videos. ``crave_score`` is produced only by the
native CRAVE model. When the CRAVE package is not installed the metric is
left unset (no heuristic/CLIP proxy).

crave_score --- higher = better quality (0-1 range)
"""

import logging
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CRAVEModule(PipelineModule):
    name = "crave"
    description = "CRAVE content-rich AIGC video evaluator (2025)"
    default_config = {
        "subsample": 12,
    }
    metric_groups = {
        "crave_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 12)
        self._ml_available = False
        self._model = None
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import crave

            self._model = crave
            self._ml_available = True
            self._backend = "crave"
            logger.info("CRAVE (native) initialised")
        except ImportError:
            logger.warning(
                "CRAVE unavailable: the native CRAVE package is not installed "
                "(github.com/littlespray/CRAVE); crave_score skipped."
            )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        try:
            score = self._predict(sample)
            if score is None:
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.crave_score = score
            logger.debug("CRAVE for %s: %.4f", sample.path.name, score)
        except Exception as e:
            logger.warning("CRAVE failed for %s: %s", sample.path, e)
        return sample

    def _predict(self, sample: Sample) -> Optional[float]:
        return float(self._model.predict(str(sample.path)))

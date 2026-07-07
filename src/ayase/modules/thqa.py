"""THQA — Talking Head Quality Assessment (ICIP 2024).

No-reference metric for evaluating talking-head video quality via the
published ``thqa`` model.

pip install thqa

thqa_score — higher = better quality.

Requires the real ``thqa`` package. If it is not installed the metric is
reported as unavailable (a previous revision fell back to a Haar-cascade
sharpness/lip-motion heuristic that does not reproduce THQA; that proxy has
been removed).
"""

import logging
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class THQAModule(PipelineModule):
    name = "thqa"
    description = "THQA talking head quality assessment (ICIP 2024)"
    default_config = {
        "subsample": 16,
    }
    metric_groups = {
        "thqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._ml_available = False
        self._backend = None
        self.subsample = self.config.get("subsample", 16)

    def setup(self) -> None:
        try:
            import thqa as thqa_lib

            self._model = thqa_lib
            self._ml_available = True
            self._backend = "thqa"
            logger.info("THQA module initialised (thqa package)")
        except ImportError:
            self._backend = "unavailable"
            logger.info(
                "THQA: 'thqa' package not installed; metric unavailable. "
                "Install with: pip install thqa"
            )
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("THQA package init failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            score = self._score_thqa_package(sample.path)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.thqa_score = score
            logger.debug(f"THQA for {sample.path.name}: {score:.4f}")
        except Exception as e:
            logger.error(f"THQA failed: {e}")
        return sample

    def _score_thqa_package(self, path: Path) -> Optional[float]:
        score = self._model.evaluate(str(path))
        return float(score) if score is not None else None

"""TTSDS2 speech quality benchmark wrapper.

TTSDS2 is a heavier TTS evaluation pipeline, so this module is opt-in
(``enabled=False`` by default). When enabled it uses an installed TTSDS2
implementation. A generic signal-based speech-quality heuristic is not
TTSDS2, so it is not emitted under the TTSDS2 name: if the ``ttsds2`` package
is not installed the module reports itself unavailable.

ttsds2_score — aggregate speech quality (0-1, higher=better), populated only
with the real TTSDS2 backend.
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TTSDS2Module(PipelineModule):
    name = "ttsds2"
    description = "TTSDS2 opt-in speech quality benchmark score"
    default_config = {
        "enabled": False,
        "sample_rate": 16000,
    }
    models = [
        {
            "id": "ttsds-benchmark",
            "type": "other",
            "task": "TTSDS2 benchmark implementation",
        },
    ]
    metric_info = {
        "ttsds2_score": "TTSDS2 aggregate speech quality score (0-1, higher=better)",
    }
    metric_groups = {
        "ttsds2_score": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.enabled = self.config.get("enabled", False)
        self.sample_rate = self.config.get("sample_rate", 16000)
        self._backend = None
        self._model = None

    def setup(self) -> None:
        if not self.enabled:
            return
        try:
            import ttsds2

            self._model = ttsds2
            self._backend = "ttsds2"
            logger.info("TTSDS2 initialised with installed package")
        except ImportError:
            self._backend = "unavailable"
            logger.info(
                "TTSDS2 unavailable: the ttsds2 package is not installed; "
                "ttsds2_score will not be populated by this module."
            )
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("TTSDS2 setup failed (%s); reporting unavailable", e)

    def process(self, sample: Sample) -> Sample:
        if not self.enabled or self._backend != "ttsds2":
            return sample
        try:
            score = self._score_package(sample.path)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.ttsds2_score = float(score)
        except Exception as e:
            logger.warning("TTSDS2 failed for %s: %s", sample.path, e)
        return sample

    def _score_package(self, path) -> Optional[float]:
        try:
            if hasattr(self._model, "score"):
                return float(self._model.score(str(path)))
            if hasattr(self._model, "evaluate"):
                result = self._model.evaluate(str(path))
                if isinstance(result, dict):
                    return float(result.get("score", result.get("overall")))
                return float(result)
        except Exception as e:
            logger.debug("TTSDS2 package scoring failed: %s", e)
        return None

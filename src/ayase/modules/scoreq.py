"""SCOREQ speech naturalness metric.

Uses SCOREQ when an implementation is available and otherwise falls back to a
bounded speech-signal quality proxy. Scores are 0-1, higher is better.
"""

import logging
from typing import Optional

from ayase.audio import load_audio, speech_quality_proxy
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SCOREQModule(PipelineModule):
    name = "scoreq"
    description = "SCOREQ no-reference speech naturalness score"
    default_config = {
        "sample_rate": 16000,
        "warning_threshold": 0.45,
    }
    models = [
        {
            "id": "alessandroragano/scoreq",
            "type": "other",
            "url": "https://github.com/alessandroragano/scoreq",
            "task": "Supervised speech naturalness scoring",
        },
    ]
    metric_info = {
        "scoreq_score": "SCOREQ speech naturalness score (0-1, higher=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.warning_threshold = self.config.get("warning_threshold", 0.45)
        self._backend = "signal_proxy"
        self._model = None

    def setup(self) -> None:
        try:
            import scoreq

            self._model = scoreq
            self._backend = "scoreq"
            logger.info("SCOREQ initialised with scoreq package")
        except ImportError:
            logger.info("SCOREQ package not installed; using signal proxy")
        except Exception as e:
            logger.warning("SCOREQ setup failed (%s); using signal proxy", e)

    def process(self, sample: Sample) -> Sample:
        try:
            score = self._score_package(sample.path) if self._backend == "scoreq" else None
            if score is None:
                audio = load_audio(sample.path, target_sr=self.sample_rate)
                if audio is None:
                    return sample
                score = speech_quality_proxy(audio, sr=self.sample_rate)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.scoreq_score = float(score)

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low SCOREQ speech quality: {score:.3f}",
                        details={"scoreq_score": float(score)},
                    )
                )
        except Exception as e:
            logger.warning("SCOREQ failed for %s: %s", sample.path, e)
        return sample

    def _score_package(self, path) -> Optional[float]:
        try:
            if hasattr(self._model, "score"):
                return float(self._model.score(str(path)))
            if hasattr(self._model, "predict"):
                return float(self._model.predict(str(path)))
        except Exception as e:
            logger.debug("SCOREQ package scoring failed: %s", e)
        return None

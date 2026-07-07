"""SCOREQ speech naturalness metric.

Uses the real SCOREQ model (alessandroragano/scoreq) in no-reference mode.
Earlier revisions fell back to a bounded speech-signal statistic when the
``scoreq`` package was missing; that proxy is not SCOREQ and has been removed.
When the model is unavailable ``scoreq_score`` is left ``None``.

SCOREQ returns a MOS-style naturalness prediction (higher = better).
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SCOREQModule(PipelineModule):
    name = "scoreq"
    description = "SCOREQ no-reference speech naturalness score"
    default_config = {
        "sample_rate": 16000,
        "data_domain": "natural",
        "warning_threshold": None,
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
        "scoreq_score": "SCOREQ speech naturalness score (MOS-style, higher=better)",
    }
    metric_groups = {
        "scoreq_score": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.data_domain = self.config.get("data_domain", "natural")
        self.warning_threshold = self.config.get("warning_threshold")
        self._backend = None
        self._model = None

    def setup(self) -> None:
        try:
            import scoreq

            self._model = scoreq.Scoreq(data_domain=self.data_domain, mode="nr")
            self._backend = "scoreq"
            logger.info("SCOREQ initialised with scoreq package (nr, %s)", self.data_domain)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("SCOREQ package not installed; scoreq_score left unset.")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("SCOREQ setup failed (%s); scoreq_score left unset.", e)

    def process(self, sample: Sample) -> Sample:
        if self._backend != "scoreq" or self._model is None:
            return sample

        try:
            score = self._model.predict(test_path=str(sample.path), ref_path=None)
            if score is None:
                return sample
            score = float(score)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.scoreq_score = score

            if self.warning_threshold is not None and score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low SCOREQ speech quality: {score:.3f}",
                        details={"scoreq_score": score},
                    )
                )
        except Exception as e:
            logger.warning("SCOREQ failed for %s: %s", sample.path, e)
        return sample

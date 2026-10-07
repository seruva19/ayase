"""Predict per-sample, no-reference speech MOS with UTMOSv2.

Audio is decoded from the sample, mixed to mono, and resampled to 16 kHz by
default. The ``utmosv2`` package provides the official model
(``utmosv2.create_model(pretrained=True)`` → ``model.predict(data=, sr=)``).
``utmos_v2_score`` is the model's predicted 1-5 speech Mean Opinion Score
(higher is better), with a warning below the configured threshold. This is a
speech-quality metric, not a general audio, caption-alignment, reference-audio,
or dataset-level metric. No proxy is emitted when the package is unavailable.
Source: https://github.com/sarulab-speech/UTMOSv2
"""

import logging
from typing import Optional

import numpy as np

from ayase.audio import load_audio
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AudioUTMOSv2Module(PipelineModule):
    name = "audio_utmos_v2"
    provenance = "published"
    sources = {
        "utmos_v2_score": "UTMOSv2 (sarulab-speech) — https://github.com/sarulab-speech/UTMOSv2",
    }
    description = "UTMOSv2 no-reference MOS prediction for speech quality"
    default_config = {
        "target_sr": 16000,
        "warning_threshold": 3.0,
    }
    models = [
        {
            "id": "sarulab-speech/UTMOSv2",
            "type": "pip_package",
            "install": "pip install git+https://github.com/sarulab-speech/UTMOSv2.git",
            "task": "UTMOSv2 speech MOS predictor",
        },
    ]
    metric_info = {
        "utmos_v2_score": "UTMOSv2 predicted speech MOS (1-5, higher=better)",
    }
    metric_groups = {
        "utmos_v2_score": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.target_sr = self.config.get("target_sr", 16000)
        self.warning_threshold = self.config.get("warning_threshold", 3.0)
        self._backend = None
        self._model = None

    def setup(self) -> None:
        try:
            import utmosv2

            self._model = utmosv2.create_model(pretrained=True)
            self._backend = "utmosv2_package"
            logger.info("UTMOSv2 initialised with utmosv2 package")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug("UTMOSv2 package setup failed: %s", e)

        self._backend = "unavailable"
        logger.warning(
            "UTMOSv2 unavailable: install the `utmosv2` package "
            "(pip install git+https://github.com/sarulab-speech/UTMOSv2.git)"
        )

    def process(self, sample: Sample) -> Sample:
        if self._backend != "utmosv2_package":
            return sample
        try:
            audio = load_audio(sample.path, target_sr=self.target_sr)
            if audio is None:
                return sample

            score = self._score_model(audio)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.utmos_v2_score = round(float(score), 3)

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low predicted MOS (UTMOSv2): {score:.2f}",
                        details={"utmos_v2_score": float(score)},
                    )
                )
        except Exception as e:
            logger.warning("UTMOSv2 failed for %s: %s", sample.path, e)
        return sample

    def _score_model(self, audio) -> Optional[float]:
        try:
            pred = self._model.predict(data=audio, sr=self.target_sr)
            return float(np.asarray(pred).reshape(-1)[0])
        except Exception as e:
            logger.debug("UTMOSv2 model scoring failed: %s", e)
        return None

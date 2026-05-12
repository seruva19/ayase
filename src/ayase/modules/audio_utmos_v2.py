"""UTMOSv2 no-reference speech MOS prediction.

Predicts speech Mean Opinion Score on a 1-5 scale. Uses UTMOSv2 when available
and a signal-quality MOS proxy otherwise, preserving graceful operation without
large model downloads.
"""

import logging
from typing import Optional

from ayase.audio import load_audio, speech_quality_proxy
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AudioUTMOSv2Module(PipelineModule):
    name = "audio_utmos_v2"
    description = "UTMOSv2 no-reference MOS prediction for speech quality"
    default_config = {
        "target_sr": 16000,
        "warning_threshold": 3.0,
        "use_torch_hub": False,
    }
    models = [
        {
            "id": "sarulab-speech/UTMOSv2",
            "type": "torch_hub",
            "task": "UTMOSv2 speech MOS predictor",
        },
    ]
    metric_info = {
        "utmos_v2_score": "UTMOSv2 predicted speech MOS (1-5, higher=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.target_sr = self.config.get("target_sr", 16000)
        self.warning_threshold = self.config.get("warning_threshold", 3.0)
        self.use_torch_hub = self.config.get("use_torch_hub", False)
        self._backend = "signal_proxy"
        self._model = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import utmosv2

            self._model = utmosv2
            self._backend = "utmosv2_package"
            logger.info("UTMOSv2 initialised with utmosv2 package")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug("UTMOSv2 package setup failed: %s", e)

        if not self.use_torch_hub:
            logger.info("UTMOSv2 torch.hub disabled; using signal proxy")
            return
        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = torch.hub.load("sarulab-speech/UTMOSv2", "utmosv2", trust_repo=True)
            self._model = self._model.to(self._device).eval()
            self._backend = "torch_hub"
            logger.info("UTMOSv2 initialised from torch.hub on %s", self._device)
        except Exception as e:
            logger.warning("UTMOSv2 setup failed (%s); using signal proxy", e)

    def process(self, sample: Sample) -> Sample:
        try:
            audio = load_audio(sample.path, target_sr=self.target_sr)
            if audio is None:
                return sample

            score = self._score_model(audio)
            if score is None:
                score = speech_quality_proxy(audio, sr=self.target_sr, mos=True)

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
            if self._backend == "utmosv2_package":
                if hasattr(self._model, "predict"):
                    return float(self._model.predict(audio, self.target_sr))
                if hasattr(self._model, "score"):
                    return float(self._model.score(audio, self.target_sr))
            if self._backend == "torch_hub":
                import torch

                waveform = torch.from_numpy(audio).float().unsqueeze(0).to(self._device)
                with torch.no_grad():
                    return float(self._model(waveform, self.target_sr).reshape(-1)[0].item())
        except Exception as e:
            logger.debug("UTMOSv2 model scoring failed: %s", e)
        return None

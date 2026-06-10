"""TTSDS2 speech quality benchmark wrapper.

TTSDS2 is a heavier TTS evaluation pipeline, so this module is opt-in by
default. When enabled, it uses an installed TTSDS2 implementation if present or
a deterministic multi-axis speech proxy for lightweight reproducibility tests.
"""

import logging
from typing import Optional

from ayase.audio import audio_feature_summary, load_audio, speech_quality_proxy
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
        self._backend = "signal_proxy"
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
            logger.info("TTSDS2 package not installed; using signal proxy")
        except Exception as e:
            logger.warning("TTSDS2 setup failed (%s); using signal proxy", e)

    def process(self, sample: Sample) -> Sample:
        if not self.enabled:
            return sample
        try:
            score = self._score_package(sample.path) if self._backend == "ttsds2" else None
            if score is None:
                audio = load_audio(sample.path, target_sr=self.sample_rate)
                if audio is None:
                    return sample
                base = speech_quality_proxy(audio, sr=self.sample_rate)
                feats = audio_feature_summary(audio, sr=self.sample_rate)
                stability = 1.0 - min(feats["spectral_flux"], 1.0)
                score = float(max(0.0, min(1.0, 0.75 * base + 0.25 * stability)))

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

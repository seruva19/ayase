"""Human-CLAP style audio-text relevance score.

Uses a configurable CLAP checkpoint, allowing fine-tuned Human-CLAP weights when
available. The default uses LAION CLAP as a practical backend. Scores are 0-1,
higher means the audio is more relevant to the caption.
"""

import logging
from typing import Optional

from ayase.audio import load_audio
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class HumanCLAPModule(PipelineModule):
    name = "human_clap"
    description = "Human-CLAP audio-text relevance score"
    default_config = {
        "model_name": "laion/clap-htsat-fused",
        "sample_rate": 48000,
        "warning_threshold": 0.25,
        "device": "auto",
    }
    models = [
        {
            "id": "laion/clap-htsat-fused",
            "type": "huggingface",
            "task": "Configurable CLAP/Human-CLAP audio-text encoder",
        },
    ]
    metric_info = {
        "human_clap_score": "Audio-text relevance from a CLAP/Human-CLAP backend (0-1)",
    }
    metric_groups = {
        "human_clap_score": "audio",
    }
    metric_field_name = "human_clap_score"

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "laion/clap-htsat-fused")
        self.sample_rate = self.config.get("sample_rate", 48000)
        self.warning_threshold = self.config.get("warning_threshold", 0.25)
        self.device_config = self.config.get("device", "auto")
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self) -> None:
        try:
            import torch
            from transformers import ClapModel, ClapProcessor

            if self.device_config == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.device_config
            models_dir = self.config.get("models_dir", "models")
            self._model = ClapModel.from_pretrained(self.model_name, cache_dir=models_dir).to(self._device)
            self._processor = ClapProcessor.from_pretrained(self.model_name, cache_dir=models_dir)
            self._model.eval()
            self._ml_available = True
            logger.info("Human-CLAP initialised with %s on %s", self.model_name, self._device)
        except ImportError:
            logger.warning("Human-CLAP requires torch and transformers")
        except Exception as e:
            logger.warning("Human-CLAP setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        caption = _caption_text(sample)
        if not caption:
            return sample
        try:
            audio = load_audio(sample.path, target_sr=self.sample_rate, duration=10.0)
            if audio is None or len(audio) == 0:
                return sample
            score = self._score(audio, caption)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            setattr(sample.quality_metrics, self.metric_field_name, float(score))

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low Human-CLAP relevance: {score:.3f}",
                        details={"human_clap_score": float(score), "caption": caption[:80]},
                    )
                )
        except Exception as e:
            logger.warning("Human-CLAP failed for %s: %s", sample.path, e)
        return sample

    def _score(self, audio, caption: str) -> Optional[float]:
        try:
            import torch

            inputs = self._processor(
                text=[caption],
                audios=[audio],
                return_tensors="pt",
                padding=True,
                sampling_rate=self.sample_rate,
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
                audio_embeds = outputs.audio_embeds
                text_embeds = outputs.text_embeds
                audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                sim = (audio_embeds * text_embeds).sum(dim=-1).item()
            return float((sim + 1.0) / 2.0)
        except Exception as e:
            logger.debug("Human-CLAP scoring failed: %s", e)
            return None


def _caption_text(sample: Sample) -> Optional[str]:
    if sample.caption and sample.caption.text:
        return sample.caption.text
    sidecar = sample.path.with_suffix(".txt")
    try:
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8").strip()
    except Exception:
        logger.debug("Failed to read caption sidecar %s", sidecar)
    return None

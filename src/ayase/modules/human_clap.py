"""Human-CLAP audio-text relevance score.

Uses the official Human-CLAP checkpoint ``sarulab-speech/human-clap-wsce-mae``
(CLAP fine-tuned on human-scored similarity) with the LAION CLAP processor,
as documented in the official repository. The score is the raw audio-text
cosine similarity (CLAPScore convention), higher means the audio is more
relevant to the caption.
"""

import logging
from typing import Optional

from ayase.audio import load_audio
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class HumanCLAPModule(PipelineModule):
    name = "human_clap"
    provenance = "published"
    sources = {
        "human_clap_score": "Human-CLAP (Takano et al., arXiv 2506.23553) — https://github.com/sarulab-speech/Human-CLAP",
    }
    description = "Human-CLAP audio-text relevance score"
    default_config = {
        "model_name": "sarulab-speech/human-clap-wsce-mae",
        "processor_name": "laion/clap-htsat-fused",
        "sample_rate": 48000,
        "warning_threshold": 0.25,
        "device": "auto",
    }
    models = [
        {
            "id": "sarulab-speech/human-clap-wsce-mae",
            "type": "huggingface",
            "task": "Human-CLAP audio-text encoder (official fine-tuned weights)",
        },
        {
            "id": "laion/clap-htsat-fused",
            "type": "huggingface",
            "task": "CLAP processor (feature extractor + tokenizer)",
        },
    ]
    metric_info = {
        "human_clap_score": "Human-CLAP audio-text cosine relevance (-1 to 1 theoretical, higher=better)",
    }
    metric_groups = {
        "human_clap_score": "audio",
    }
    metric_field_name = "human_clap_score"

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "sarulab-speech/human-clap-wsce-mae")
        # Official usage loads the model from the Human-CLAP repo and the
        # processor from the LAION CLAP repo it was fine-tuned from.
        self.processor_name = self.config.get("processor_name", "laion/clap-htsat-fused")
        self.sample_rate = self.config.get("sample_rate", 48000)
        self.warning_threshold = self.config.get("warning_threshold", 0.25)
        self.device_config = self.config.get("device", "auto")
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            from transformers import ClapModel, ClapProcessor
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.device_config)
            models_dir = self.config.get("models_dir", "models")
            self._model = ClapModel.from_pretrained(self.model_name, cache_dir=models_dir).to(self._device)
            self._processor = ClapProcessor.from_pretrained(self.processor_name, cache_dir=models_dir)
            self._model.eval()
            self._ml_available = True
            self._backend = "clap"
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
            # Human-CLAP scores the whole clip; the CLAP processor handles
            # windowing internally — no fixed 10 s truncation.
            audio = load_audio(sample.path, target_sr=self.sample_rate, duration=None)
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
            # Raw cosine — the CLAPScore convention used by Human-CLAP.
            return float(sim)
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

"""AQAScore audio question-answering alignment.

Heavy Qwen2.5-Omni based audio QA is opt-in by default. When enabled without a
local Omni backend, the module reports a conservative audio/caption proxy so the
pipeline remains deterministic and non-failing.
"""

import logging
from typing import Optional

from ayase.audio import load_audio, speech_quality_proxy
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AQAScoreModule(PipelineModule):
    name = "aqascore"
    description = "AQAScore opt-in audio question-answering alignment"
    default_config = {
        "enabled": False,
        "backend": "auto",  # auto | proxy
        "model_name": "Qwen/Qwen2.5-Omni-7B",
        "sample_rate": 16000,
        "device": "auto",
    }
    models = [
        {
            "id": "Qwen/Qwen2.5-Omni-7B",
            "type": "huggingface",
            "task": "Optional audio question-answering evaluator",
            "notes": "Heavy opt-in backend; default module config leaves it disabled.",
        },
    ]
    metric_info = {
        "aqascore_score": "Audio question-answering alignment score (0-1, higher=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.enabled = self.config.get("enabled", False)
        self.backend = str(self.config.get("backend", "auto")).lower()
        self.model_name = self.config.get("model_name", "Qwen/Qwen2.5-Omni-7B")
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.device_config = self.config.get("device", "auto")
        self._backend = "caption_proxy"
        self._model = None
        self._processor = None
        self._device = "cpu"

    def setup(self) -> None:
        if not self.enabled:
            return
        if self.backend == "proxy":
            logger.info("AQAScore: using deterministic audio/caption proxy backend")
            return
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
            from ayase.runtime import from_pretrained_with_attention, resolve_torch_device

            model_cls = getattr(transformers, "Qwen2_5OmniForConditionalGeneration", None)
            if model_cls is None:
                raise ImportError("Qwen2_5OmniForConditionalGeneration unavailable")
            self._device = resolve_torch_device(self.device_config)
            models_dir = self.config.get("models_dir", "models")
            self._processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=models_dir)
            self._model = from_pretrained_with_attention(
                model_cls,
                self.model_name,
                self.config,
                device=self._device,
                cache_dir=models_dir,
                torch_dtype="auto",
                device_map="auto" if self._device == "cuda" else None,
            )
            self._backend = "qwen_omni"
            logger.info("AQAScore initialised with %s", self.model_name)
        except ImportError:
            logger.info("AQAScore Omni backend unavailable; using caption proxy")
        except Exception as e:
            logger.warning("AQAScore setup failed (%s); using caption proxy", e)

    def process(self, sample: Sample) -> Sample:
        if not self.enabled:
            return sample
        caption = _caption_text(sample)
        if not caption:
            return sample
        try:
            audio = load_audio(sample.path, target_sr=self.sample_rate, duration=20.0)
            if audio is None or len(audio) == 0:
                return sample

            score = self._score_qwen(sample.path, caption) if self._backend == "qwen_omni" else None
            if score is None:
                score = self._score_proxy(audio, caption)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.aqascore_score = float(score)
        except Exception as e:
            logger.warning("AQAScore failed for %s: %s", sample.path, e)
        return sample

    def _score_qwen(self, path, caption: str) -> Optional[float]:
        try:
            prompt = (
                "Listen to the audio and answer whether it matches this description. "
                f"Description: {caption}. Reply with a score from 0 to 1."
            )
            messages = [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(path)},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=text, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._device)
            outputs = self._model.generate(**inputs, max_new_tokens=16)
            decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return _extract_score(decoded)
        except Exception as e:
            logger.debug("AQAScore Omni scoring failed: %s", e)
            return None

    def _score_proxy(self, audio, caption: str) -> float:
        quality = speech_quality_proxy(audio, sr=self.sample_rate)
        caption_has_audio_terms = any(
            word in caption.lower()
            for word in ("speech", "voice", "music", "sound", "audio", "sing", "noise", "talk")
        )
        return float(min(1.0, quality * (1.0 if caption_has_audio_terms else 0.75)))


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


def _extract_score(text: str) -> Optional[float]:
    import re

    matches = re.findall(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text)
    if not matches:
        return None
    return float(max(0.0, min(1.0, float(matches[-1]))))

"""AQAScore — probabilistic semantic verification of audio by an ALLM.

Implements the AQAScore mechanism (Kuan, Chang, Lee, arXiv:2601.14728): the
audio and the description are passed to an audio-aware LLM
(``Qwen/Qwen2.5-Omni-7B``) with the paper's binary query template
"Does this audio contain the sound events described by the text: {desc}", and
the score is the exact first-token probability P("Yes"), normalized over the
{"Yes", "No"} token pair. Higher means stronger audio-text semantic
alignment. Single-query form (the paper decomposes captions into several
targeted queries); the module is opt-in via ``enabled: true`` and emits no
score when the backend is unavailable. There is no proxy path.
"""

import logging
from typing import Optional

from ayase.audio import load_audio
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AQAScoreModule(PipelineModule):
    name = "aqascore"
    provenance = "published"
    sources = {
        "aqascore_score": "AQAScore (Kuan, Chang, Lee, 2026) — https://arxiv.org/abs/2601.14728",
    }
    description = "AQAScore opt-in audio question-answering alignment (P(Yes) protocol)"
    default_config = {
        "enabled": False,
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
    metric_groups = {
        "aqascore_score": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.enabled = self.config.get("enabled", False)
        self.model_name = self.config.get("model_name", "Qwen/Qwen2.5-Omni-7B")
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.device_config = self.config.get("device", "auto")
        self._backend = "unavailable"
        self._model = None
        self._processor = None
        self._device = "cpu"

    def setup(self) -> None:
        if not self.enabled:
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
            logger.warning(
                "AQAScore unavailable: Qwen2.5-Omni backend not installed; "
                "aqascore_score will be left unset."
            )
        except Exception as e:
            logger.warning("AQAScore unavailable: setup failed (%s)", e)

    def process(self, sample: Sample) -> Sample:
        if not self.enabled or self._backend != "qwen_omni":
            return sample
        caption = _caption_text(sample)
        if not caption:
            return sample
        try:
            audio = load_audio(sample.path, target_sr=self.sample_rate, duration=20.0)
            if audio is None or len(audio) == 0:
                return sample

            score = self._score_qwen(sample.path, caption, audio)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.aqascore_score = float(score)
        except Exception as e:
            logger.warning("AQAScore failed for %s: %s", sample.path, e)
        return sample

    def _score_qwen(self, path, caption: str, audio) -> Optional[float]:
        """P("Yes") from the first generated token — the AQAScore mechanism."""
        try:
            import torch

            # The paper's query template for a single semantic query.
            prompt = (
                "Does this audio contain the sound events described by the "
                f'text: "{caption}" Answer only Yes or No.'
            )
            messages = [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(path)},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=text,
                audio=[audio],
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            # logits of the first generated token -> softmax over Yes/No
            first_logits = outputs.scores[0][0].float()
            tokenizer = getattr(self._processor, "tokenizer", None) or getattr(
                self._processor, "text_tokenizer", None
            )
            yes_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
            no_id = tokenizer("No", add_special_tokens=False)["input_ids"][0]
            pair = torch.stack([first_logits[yes_id], first_logits[no_id]])
            return float(torch.softmax(pair, dim=0)[0].item())
        except Exception as e:
            logger.debug("AQAScore Omni scoring failed: %s", e)
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

"""UnifiedReward 2.0 prompt-image reward with alignment, coherence, and style scores."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, Optional

from PIL import Image

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

from ._reward_utils import get_prompt, image_to_data_url, load_rgb_image, post_openai_chat

logger = logging.getLogger(__name__)


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+", value)
        if match:
            return float(match.group())
    return None


def _mean_non_none(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _extract_labeled_number(text: str, label: str) -> Optional[float]:
    cleaned = (text or "").replace("*", "")
    pattern = rf"{re.escape(label)}\s*(?:\([^)]+\))?\s*[::]\s*([-+]?\d*\.?\d+)"
    match = re.search(pattern, cleaned, flags=re.I)
    return _coerce_float(match.group(1)) if match else None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        return None
    payload = match.group(0)
    for loader in (json.loads, lambda s: json.loads(s.replace("'", '"'))):
        try:
            parsed = loader(payload)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
    return None


def _score_from_json(parsed: Dict[str, Any], *names: str) -> Optional[float]:
    normalized = {re.sub(r"[^a-z0-9]+", "", str(k).lower()): v for k, v in parsed.items()}
    for name in names:
        key = re.sub(r"[^a-z0-9]+", "", name.lower())
        value = normalized.get(key)
        if isinstance(value, dict):
            value = value.get("score") or value.get("value")
        score = _coerce_float(value)
        if score is not None:
            return score
    return None


def parse_unified_reward_2_output(text: str) -> Dict[str, Optional[float]]:
    parsed = _extract_json_object(text)
    if parsed:
        alignment = _score_from_json(parsed, "alignment", "alignment score")
        coherence = _score_from_json(parsed, "coherence", "coherence score")
        style = _score_from_json(parsed, "style", "style score")
    else:
        alignment = coherence = style = None

    if alignment is None:
        alignment = _extract_labeled_number(text, "Alignment Score")
    if coherence is None:
        coherence = _extract_labeled_number(text, "Coherence Score")
    if style is None:
        style = _extract_labeled_number(text, "Style Score")

    return {
        "alignment": alignment,
        "coherence": coherence,
        "style": style,
        "score": _mean_non_none([alignment, coherence, style]),
    }


def build_unified_reward_2_prompt(prompt: Optional[str]) -> str:
    prompt = prompt or ""
    return (
        "You are presented with a generated image and its associated text caption. "
        "Your task is to analyze the image across multiple dimensions in relation to the caption. "
        "Specifically:\n"
        "Provide overall assessments for the image along the following axes "
        "(each rated from 1 to 5):\n"
        "- Alignment Score: How well the image matches the caption in terms of content.\n"
        "- Coherence Score: How logically consistent the image is "
        "(absence of visual glitches, object distortions, etc.).\n"
        "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
        "Output your evaluation using the format below:\n\n"
        "Alignment Score (1-5): X\n"
        "Coherence Score (1-5): Y\n"
        "Style Score (1-5): Z\n\n"
        "Do not include explanations, analysis, bullet points, or any text outside "
        "the requested output format.\n\n"
        "Your task is provided as follows:\n"
        f"Text Caption: [{prompt}]"
    )


class UnifiedReward2Module(PipelineModule):
    name = "unified_reward_2"
    description = "UnifiedReward 2.0 multi-dimensional prompt-image reward scoring"
    default_config = {
        "backend": "auto",  # auto | diffsynth | openai
        "model_name": "UnifiedReward-2.0-qwen35-9b",
        "endpoint_url": None,
        "api_key": None,
        "prompt": None,
        "device": "auto",
        "dtype": "bfloat16",
        "max_new_tokens": 1024,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_image_size": 1024,
        "resize_to_square": False,
        "warning_threshold": None,
        "vram_limit": None,
        "store_raw_outputs": False,
    }
    models = [
        {
            "id": "DiffSynth-Studio/ImageMetrics:UnifiedReward-2.0-qwen35-9b",
            "type": "other",
            "url": "https://modelscope.cn/models/DiffSynth-Studio/ImageMetrics",
            "task": "UnifiedReward 2.0 Qwen3.5-VL reward model",
            "size": "9B",
            "notes": "Loaded through optional DiffSynth or served through an endpoint.",
        },
    ]
    metric_info = {
        "unified_reward_2_score": "UnifiedReward 2.0 mean score (1-5, higher=better)",
        "unified_reward_2_alignment_score": "UnifiedReward 2.0 alignment score (1-5)",
        "unified_reward_2_coherence_score": "UnifiedReward 2.0 coherence score (1-5)",
        "unified_reward_2_style_score": "UnifiedReward 2.0 style score (1-5)",
    }
    metric_groups = {
        "unified_reward_2_score": "nr_quality",
        "unified_reward_2_alignment_score": "alignment",
        "unified_reward_2_coherence_score": "nr_quality",
        "unified_reward_2_style_score": "aesthetic",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.backend_pref = str(self.config.get("backend", "auto")).lower()
        self.model_name = self.config.get("model_name", "UnifiedReward-2.0-qwen35-9b")
        self.endpoint_url = self.config.get("endpoint_url")
        self.api_key = self.config.get("api_key")
        self.device_config = self.config.get("device", "auto")
        self.dtype_config = self.config.get("dtype", "bfloat16")
        self.max_new_tokens = int(self.config.get("max_new_tokens", 1024))
        self.max_image_size = int(self.config.get("max_image_size", 1024))
        self.resize_to_square = bool(self.config.get("resize_to_square", False))
        self.warning_threshold = self.config.get("warning_threshold")
        self._backend = None
        self._model = None
        self._ml_available = False

    def setup(self) -> None:
        if self.endpoint_url and self.backend_pref in ("auto", "openai", "endpoint"):
            self._backend = "openai"
            self._ml_available = True
            return

        if self.backend_pref in ("openai", "endpoint"):
            logger.warning("UnifiedReward 2.0 endpoint backend requires endpoint_url.")
            return

        if self.backend_pref in ("auto", "diffsynth"):
            self._try_diffsynth()

    def process(self, sample: Sample) -> Sample:
        if sample.is_video or not self._ml_available:
            return sample

        prompt = get_prompt(sample, self.config)
        if not prompt:
            return sample

        try:
            image = load_rgb_image(
                sample.path,
                max_image_size=self.max_image_size,
                resize_to_square=self.resize_to_square,
            )
            if image is None:
                return sample

            parsed, raw_output = self._score_image(prompt, image)
            if parsed.get("score") is None:
                return sample

            self._store_scores(sample, parsed)
            self._store_details(sample, parsed, raw_output)

            score = parsed.get("score")
            if (
                score is not None
                and self.warning_threshold is not None
                and score < self.warning_threshold
            ):
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low UnifiedReward 2.0 score: {score:.3f}",
                        details={"unified_reward_2_score": score},
                    )
                )
        except Exception as e:
            logger.warning("UnifiedReward 2.0 failed for %s: %s", sample.path, e)

        return sample

    def _try_diffsynth(self) -> bool:
        try:
            import torch
            from diffsynth.metrics import UnifiedReward2Metric

            from ayase.runtime import resolve_torch_device, resolve_torch_dtype

            device = resolve_torch_device(self.device_config)
            dtype = resolve_torch_dtype(device, self.dtype_config)
            self._model = UnifiedReward2Metric.from_pretrained(
                torch_dtype=dtype or torch.bfloat16,
                device=torch.device(device),
                max_new_tokens=self.max_new_tokens,
                vram_limit=self.config.get("vram_limit"),
            )
            self._backend = "diffsynth"
            self._ml_available = True
            logger.info("DiffSynth UnifiedReward 2.0 backend initialized on %s", device)
            return True
        except Exception as e:
            logger.warning("UnifiedReward 2.0 unavailable: %s", e)
            return False

    def _score_image(
        self,
        prompt: str,
        image: Image.Image,
    ) -> tuple[Dict[str, Optional[float]], Optional[str]]:
        if self._backend == "diffsynth":
            outputs = self._model.evaluate(prompt, image)
            parsed = outputs[0] if outputs else {}
            return {
                "alignment": _coerce_float(parsed.get("alignment")),
                "coherence": _coerce_float(parsed.get("coherence")),
                "style": _coerce_float(parsed.get("style")),
                "score": _coerce_float(parsed.get("score")),
            }, None

        raw_output = self._generate_openai(prompt, image)
        return parse_unified_reward_2_output(raw_output), raw_output

    def _generate_openai(self, prompt: str, image: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                    {"type": "text", "text": build_unified_reward_2_prompt(prompt)},
                ],
            }
        ]
        return post_openai_chat(
            endpoint_url=self.endpoint_url,
            api_key=self.api_key,
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=float(self.config.get("temperature", 0.0)),
            top_p=float(self.config.get("top_p", 1.0)),
            timeout=float(self.config.get("timeout", 600)),
        )

    @staticmethod
    def _store_scores(sample: Sample, parsed: Dict[str, Optional[float]]) -> None:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        qm = sample.quality_metrics
        qm.unified_reward_2_alignment_score = parsed.get("alignment")
        qm.unified_reward_2_coherence_score = parsed.get("coherence")
        qm.unified_reward_2_style_score = parsed.get("style")
        qm.unified_reward_2_score = parsed.get("score")

    def _store_details(
        self,
        sample: Sample,
        parsed: Dict[str, Optional[float]],
        raw_output: Optional[str],
    ) -> None:
        sample.detections.append(
            {
                "type": "unified_reward_2",
                "backend": self._backend,
                "scores": parsed,
                "raw_output": raw_output if self.config.get("store_raw_outputs", False) else None,
            }
        )

"""EvoQuality self-evolving VLM no-reference image quality scoring.

Scores images with ByteDance EvoQuality (Qwen2.5-VL-7B refined via majority
voting and GRPO, ICLR 2026). The model rates overall picture quality on a
1-5 scale and puts the final answer in ``\\boxed{}``. Videos are scored by
uniformly sampling frames and averaging frame scores.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

from ._reward_utils import image_to_data_url, load_rgb_image, post_openai_chat

logger = logging.getLogger(__name__)

EVOQUALITY_PROMPT = (
    "You are doing the image quality assessment task. Here is the question: "
    "What is your overall rating on the quality of this picture? "
    "The rating should be a float between 1 and 5, rounded to two decimal places, "
    "with 1 representing very poor quality and 5 representing excellent quality. "
    "You FIRST think about the reasoning process as an internal monologue and "
    "then provide the final answer. The final answer MUST BE put in \\boxed{}."
)


def parse_evoquality_output(text: str) -> Optional[float]:
    """Extract the 1-5 quality rating from EvoQuality model output."""
    for boxed in reversed(re.findall(r"boxed\{([^{}]*)\}", text or "")):
        match = re.search(r"[-+]?\d*\.?\d+", boxed)
        if match:
            return min(5.0, max(1.0, float(match.group())))

    for candidate in reversed(re.findall(r"[-+]?\d*\.?\d+", text or "")):
        try:
            value = float(candidate)
        except ValueError:
            continue
        if 1.0 <= value <= 5.0:
            return value
    return None


class EvoQualityModule(PipelineModule):
    name = "evoquality"
    description = "EvoQuality self-evolving VLM no-reference quality rating"
    default_config = {
        "backend": "auto",  # auto | transformers | openai
        "model_name": "ByteDance/EvoQuality",
        "endpoint_url": None,
        "api_key": None,
        "num_frames": 5,
        "device": "auto",
        "dtype": "bfloat16",
        "max_new_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_image_size": 1024,
        "resize_to_square": False,
        "warning_threshold": None,
        "store_raw_outputs": False,
    }
    models = [
        {
            "id": "ByteDance/EvoQuality",
            "type": "huggingface",
            "task": "Self-evolving Qwen2.5-VL NR-IQA rating model",
            "size": "7B",
            "notes": "Loaded through transformers or served through an endpoint.",
        },
    ]
    metric_info = {
        "evoquality_score": "EvoQuality VLM quality rating (1-5, higher=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.backend_pref = str(self.config.get("backend", "auto")).lower()
        self.model_name = self.config.get("model_name", "ByteDance/EvoQuality")
        self.endpoint_url = self.config.get("endpoint_url")
        self.api_key = self.config.get("api_key")
        self.num_frames = int(self.config.get("num_frames", 5))
        self.device_config = self.config.get("device", "auto")
        self.dtype_config = self.config.get("dtype", "bfloat16")
        self.max_new_tokens = int(self.config.get("max_new_tokens", 512))
        self.max_image_size = int(self.config.get("max_image_size", 1024))
        self.resize_to_square = bool(self.config.get("resize_to_square", False))
        self.warning_threshold = self.config.get("warning_threshold")
        self._backend = None
        self._model = None
        self._processor = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return

        if self.endpoint_url and self.backend_pref in ("auto", "openai", "endpoint"):
            self._backend = "openai"
            self._ml_available = True
            return

        if self.backend_pref in ("openai", "endpoint"):
            logger.warning("EvoQuality endpoint backend requires endpoint_url.")
            return

        if self.backend_pref in ("auto", "transformers"):
            self._try_transformers()
        elif self.backend_pref != "auto":
            logger.warning("Unsupported EvoQuality backend: %s", self.backend_pref)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            score, raw_outputs = self._score_frames(frames)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.evoquality_score = score

            sample.detections.append(
                {
                    "type": "evoquality",
                    "backend": self._backend,
                    "score": score,
                    "raw_outputs": raw_outputs
                    if self.config.get("store_raw_outputs", False)
                    else None,
                }
            )

            if self.warning_threshold is not None and score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low EvoQuality score: {score:.2f}",
                        details={"evoquality_score": score},
                    )
                )
        except Exception as e:
            logger.warning("EvoQuality inference failed for %s: %s", sample.path, e)

        return sample

    def _try_transformers(self) -> bool:
        try:
            import torch
            from transformers import AutoProcessor

            from ayase.runtime import resolve_torch_device, resolve_torch_dtype

            try:
                from transformers import AutoModelForImageTextToText as model_cls
            except ImportError:
                from transformers import Qwen2_5_VLForConditionalGeneration as model_cls

            device = resolve_torch_device(self.device_config)
            dtype = resolve_torch_dtype(device, self.dtype_config)

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = (
                model_cls.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype or torch.bfloat16,
                )
                .to(torch.device(device))
                .eval()
            )
            self._backend = "transformers"
            self._ml_available = True
            logger.info("EvoQuality transformers backend initialized on %s", device)
            return True
        except Exception as e:
            logger.warning("EvoQuality unavailable: %s", e)
            return False

    def _score_frames(
        self, frames: List[Image.Image]
    ) -> Tuple[Optional[float], List[str]]:
        scores = []
        raw_outputs = []
        for frame in frames:
            try:
                if self._backend == "openai":
                    raw = self._generate_openai(frame)
                else:
                    raw = self._generate_transformers(frame)
                raw_outputs.append(raw)
                score = parse_evoquality_output(raw)
                if score is not None:
                    scores.append(score)
            except Exception as e:
                logger.debug("EvoQuality frame scoring failed: %s", e)

        return (float(np.mean(scores)) if scores else None), raw_outputs

    def _generate_transformers(self, image: Image.Image) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": EVOQUALITY_PROMPT},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(
            self._model.device
        )
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0]

    def _generate_openai(self, image: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                    {"type": "text", "text": EVOQUALITY_PROMPT},
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

    def _load_frames(self, sample: Sample) -> List[Image.Image]:
        if not sample.is_video:
            image = load_rgb_image(
                sample.path,
                max_image_size=self.max_image_size,
                resize_to_square=self.resize_to_square,
            )
            return [image] if image is not None else []

        try:
            arrays = sample_frames(sample.path, max_frames=self.num_frames, color="rgb")
            return arrays_to_pil(arrays)
        except Exception as e:
            logger.debug("EvoQuality frame loading failed for %s: %s", sample.path, e)
            return []

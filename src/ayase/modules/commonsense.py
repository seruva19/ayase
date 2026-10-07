"""Estimate per-sample visual plausibility from one representative frame.

The preferred ``llava-hf/llava-1.5-7b-hf`` backend receives a fixed English
prompt asking for 1-5 ratings of location plausibility, interaction sense, and
layout consistency; their sum is divided by 15 and clipped to [0, 1], with
higher values reflecting the model's plausibility judgment. When LLaVA cannot
be loaded the module emits no score — a different-quantity fallback
(a yes/no-answer fraction from a fixed question set) is not a substitute for
the rubric. The module uses no sample caption, generation prompt,
reference media, temporal evidence, or dataset aggregation; model and language
limitations follow the selected checkpoint.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.image import load_representative_frame
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CommonsenseModule(PipelineModule):
    name = "commonsense"
    deprecated = True
    provenance = "own"
    description = "Common sense adherence (LLaVA VLM rubric)"
    default_config = {
        "vlm_model": "llava-hf/llava-1.5-7b-hf",
    }
    metric_groups = {
        "commonsense_score": "scene",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._vlm_model = None
        self._vlm_processor = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: LLaVA VLM
        try:
            import torch
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            from ayase.runtime import from_pretrained_with_attention, resolve_torch_device

            vlm_name = self.config.get("vlm_model", "llava-hf/llava-1.5-7b-hf")
            self._device = resolve_torch_device(self.config.get("device", "auto"))
            models_dir = self.config.get("models_dir", "models")
            dtype = torch.float16 if self._device == "cuda" else torch.float32

            self._vlm_model = from_pretrained_with_attention(
                LlavaNextForConditionalGeneration,
                vlm_name,
                self.config,
                device=self._device,
                torch_dtype=dtype,
                cache_dir=models_dir,
                low_cpu_mem_usage=True,
            ).to(self._device)
            self._vlm_model.eval()
            self._vlm_processor = LlavaNextProcessor.from_pretrained(vlm_name, cache_dir=models_dir)
            self._backend = "vlm"
            self._ml_available = True
            logger.info("Commonsense loaded LLaVA on %s", self._device)
            return
        except Exception as e:
            logger.info("VLM unavailable for commonsense: %s", e)

        self._backend = "unavailable"
        logger.warning("Commonsense unavailable: LLaVA VLM could not be loaded")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        image = self._load_image(sample)
        if image is None:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            if self._backend == "vlm":
                score, issues = self._compute_vlm(image)
            else:
                return sample

            if score is not None:
                sample.quality_metrics.commonsense_score = score

            for issue in issues:
                sample.validation_issues.append(issue)

        except Exception as e:
            logger.warning("Commonsense check failed: %s", e)

        return sample

    # ------------------------------------------------------------------ #
    # Tier 1: VLM (LLaVA)                                                 #
    # ------------------------------------------------------------------ #

    def _compute_vlm(self, image: np.ndarray) -> tuple:
        import torch
        import json
        import re
        from PIL import Image

        issues = []
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        prompt = (
            "USER: <image>\nRate the following aspects of this image on a scale of 1-5:\n"
            "1. Location plausibility (do objects appear in plausible locations?)\n"
            "2. Interaction sense (do interactions between objects/people make sense?)\n"
            "3. Layout consistency (is the spatial layout logically consistent?)\n"
            "Respond ONLY with a JSON object: {\"location\": N, \"interaction\": N, \"layout\": N}\n"
            "ASSISTANT:"
        )

        inputs = self._vlm_processor(prompt, images=pil_image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._vlm_model.generate(**inputs, max_new_tokens=64)
            response = self._vlm_processor.decode(output[0], skip_special_tokens=True)

        response_clean = response.split("ASSISTANT:")[-1].strip()

        try:
            json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group(0))
                loc = float(scores.get("location", 3))
                inter = float(scores.get("interaction", 3))
                layout = float(scores.get("layout", 3))
                # Normalize 1-5 -> 0-1
                score = (loc + inter + layout) / 15.0
                return float(np.clip(score, 0.0, 1.0)), issues
        except (json.JSONDecodeError, ValueError):
            pass

        # Unparseable model response — do not fabricate a score.
        return None, issues

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            return load_representative_frame(sample.path, color="bgr")
        except Exception:
            return None

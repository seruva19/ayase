"""Estimate visual novelty and composition from one representative frame.

The preferred ``llava-hf/llava-1.5-7b-hf`` backend receives a fixed English
prompt asking for 1-5 novelty, composition, and imagination ratings; their sum
is divided by 15 and clipped to [0, 1]. When LLaVA cannot be loaded the module
emits no score — a handcrafted CLIP-distance heuristic is a different,
uncalibrated quantity and is not substituted under the same field.
``creativity_score`` is per-sample and higher means more creativity under the
rubric. No sample caption, generation prompt, reference media, temporal
evidence, or dataset context is used. Applicability follows the LLaVA model
domain rather than an objective creativity ground truth.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.image import load_representative_frame
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CreativityModule(PipelineModule):
    name = "creativity"
    deprecated = True
    provenance = "own"
    description = "Artistic novelty assessment (LLaVA VLM rubric)"
    default_config = {
        "vlm_model": "llava-hf/llava-1.5-7b-hf",
    }
    metric_groups = {
        "creativity_score": "aesthetic",
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
            logger.info("Creativity loaded LLaVA on %s", self._device)
            return
        except Exception as e:
            logger.info("VLM unavailable for creativity: %s", e)

        self._backend = "unavailable"
        logger.warning("Creativity unavailable: LLaVA VLM could not be loaded")

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
                score = self._compute_vlm(image)
            else:
                return sample

            if score is not None:
                sample.quality_metrics.creativity_score = score

        except Exception as e:
            logger.warning("Creativity check failed: %s", e)

        return sample

    # ------------------------------------------------------------------ #
    # Tier 1: VLM (LLaVA)                                                 #
    # ------------------------------------------------------------------ #

    def _compute_vlm(self, image: np.ndarray) -> Optional[float]:
        import torch
        import json
        import re
        from PIL import Image

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        prompt = (
            "USER: <image>\nRate the following aspects of this image on a scale of 1-5:\n"
            "1. Visual novelty (how unusual or surprising is the visual content?)\n"
            "2. Artistic composition (how creative is the framing, color, and arrangement?)\n"
            "3. Imaginative interpretation (how much creative liberty is shown?)\n"
            "Respond ONLY with a JSON object: {\"novelty\": N, \"composition\": N, \"imagination\": N}\n"
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
                nov = float(scores.get("novelty", 3))
                comp = float(scores.get("composition", 3))
                imag = float(scores.get("imagination", 3))
                score = (nov + comp + imag) / 15.0
                return float(np.clip(score, 0.0, 1.0))
        except (json.JSONDecodeError, ValueError):
            pass

        # Unparseable model response — do not fabricate a score.
        return None

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            return load_representative_frame(sample.path, color="bgr")
        except Exception:
            return None

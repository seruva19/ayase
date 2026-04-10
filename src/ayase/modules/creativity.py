"""Creativity module — VBench-2.0 dimension.

Assesses artistic novelty and creative interpretation quality.

Backend tiers:
  1. **VLM** — LLaVA-1.5-7b with creativity assessment prompt
  2. **CLIP** — CLIP novelty distance + LAION aesthetic score
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.compat import extract_features

logger = logging.getLogger(__name__)

# Common prompt embeddings for CLIP novelty baseline
_COMMON_PROMPTS = [
    "a photo of a person",
    "a photo of a landscape",
    "a photo of a building",
    "a photo of an animal",
    "a photo of a car",
    "a photo of food on a plate",
    "a photo of a street scene",
    "a photo of the sky",
    "a photo of a room interior",
    "a photo of a group of people",
]


class CreativityModule(PipelineModule):
    name = "creativity"
    description = "Artistic novelty assessment (VLM / CLIP)"
    default_config = {
        "vlm_model": "llava-hf/llava-1.5-7b-hf",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._vlm_model = None
        self._vlm_processor = None
        self._clip_model = None
        self._clip_processor = None
        self._common_embeddings = None
        self._aes_model = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: LLaVA VLM
        try:
            import torch
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

            vlm_name = self.config.get("vlm_model", "llava-hf/llava-1.5-7b-hf")
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            models_dir = self.config.get("models_dir", "models")
            dtype = torch.float16 if self._device == "cuda" else torch.float32

            self._vlm_model = LlavaNextForConditionalGeneration.from_pretrained(
                vlm_name, torch_dtype=dtype, cache_dir=models_dir, low_cpu_mem_usage=True,
            ).to(self._device)
            self._vlm_model.eval()
            self._vlm_processor = LlavaNextProcessor.from_pretrained(vlm_name, cache_dir=models_dir)
            self._backend = "vlm"
            self._ml_available = True
            logger.info("Creativity loaded LLaVA on %s", self._device)
            return
        except Exception as e:
            logger.info("VLM unavailable for creativity: %s", e)

        # Tier 2: CLIP + aesthetic
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            models_dir = self.config.get("models_dir", "models")

            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir=models_dir,
            ).to(self._device)
            self._clip_model.eval()
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir=models_dir,
            )

            # Pre-compute common prompt embeddings
            with torch.no_grad():
                inputs = self._clip_processor(text=_COMMON_PROMPTS, return_tensors="pt", padding=True).to(self._device)
                text_features = extract_features(self._clip_model.get_text_features(**inputs))
                self._common_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)

            # Pre-load LAION aesthetic model for reuse
            try:
                import pyiqa
                self._aes_model = pyiqa.create_metric("laion_aes", device=self._device)
            except Exception:
                pass

            self._backend = "clip"
            self._ml_available = True
            logger.info("Creativity loaded CLIP on %s", self._device)
            return
        except Exception as e:
            logger.info("CLIP unavailable for creativity: %s", e)

        logger.warning("Creativity unavailable: install transformers")

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
            elif self._backend == "clip":
                score = self._compute_clip(image)
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

        return 0.5

    # ------------------------------------------------------------------ #
    # Tier 2: CLIP novelty + aesthetic                                     #
    # ------------------------------------------------------------------ #

    def _compute_clip(self, image: np.ndarray) -> Optional[float]:
        import torch
        from PIL import Image

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        with torch.no_grad():
            inputs = self._clip_processor(images=pil_image, return_tensors="pt").to(self._device)
            image_features = extract_features(self._clip_model.get_image_features(**inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Novelty = distance from common prompts
            similarities = (image_features @ self._common_embeddings.T).squeeze()
            max_sim = float(similarities.max())
            # High distance from common = high novelty
            novelty = 1.0 - max_sim  # CLIP sim is typically 0.1-0.4

        # Try LAION aesthetic via cached pyiqa model
        aesthetic_score = 0.5
        try:
            if self._aes_model is not None:
                import torchvision.transforms.functional as TF
                img_tensor = TF.to_tensor(pil_image).unsqueeze(0).to(self._device)
                raw = float(self._aes_model(img_tensor).item())
                aesthetic_score = min(raw / 10.0, 1.0)  # LAION is 0-10
        except Exception:
            pass

        # Normalize novelty to reasonable range
        novelty_normalized = min(max(novelty * 2.0, 0.0), 1.0)
        score = 0.6 * novelty_normalized + 0.4 * aesthetic_score
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
                ret, frame = cap.read()
                cap.release()
                return frame if ret else None
            else:
                return cv2.imread(str(sample.path))
        except Exception:
            return None

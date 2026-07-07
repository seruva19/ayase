"""Zero-shot scene context tagging using CLIP with predefined label candidates.

Classifies images into scene categories (outdoors, nature, urban, etc.) via
CLIP zero-shot prediction. Returns top-3 tags with confidence scores."""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.image import load_representative_frame
from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SceneTaggingModule(PipelineModule):
    name = "scene_tagging"
    description = "Zero-shot scene context tags via CLIP (top-3 scene labels)"
    default_config = {"models_dir": "models"}

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False

        # A small set of common scene tags to check against
        self.candidate_labels = [
            "outdoors",
            "indoors",
            "nature",
            "urban",
            "people",
            "animals",
            "daylight",
            "night",
            "water",
            "forest",
            "street",
            "office",
            "home",
        ]

    def setup(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            logger.info(f"Loading CLIP for Scene Tagging on {self._device}...")

            models_dir = self.config.get("models_dir", "models")

            model_name = "openai/clip-vit-base-patch32"
            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    model_name,
                    self.config,
                    device=self._device,
                    cache_dir=models_dir,
                    use_safetensors=True,
                ).to(self._device).eval()
                processor = CLIPProcessor.from_pretrained(model_name, cache_dir=models_dir)
                return model, processor

            self._model, self._processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    model_name,
                    self._device,
                    str(self.config.get("attention_backend", "auto")),
                    "safetensors",
                ),
                load_clip,
            )
            self._ml_available = True

        except ImportError:
            logger.warning("Transformers not installed. Scene tagging disabled.")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        image = self._load_image(sample)
        if image is None:
            return sample

        try:
            import torch
            from PIL import Image

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Zero-shot classification
            inputs = self._processor(
                text=self.candidate_labels, images=pil_image, return_tensors="pt", padding=True
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # Get top tags
            top_probs, top_indices = probs.topk(3)
            detected_tags = [self.candidate_labels[idx] for idx in top_indices[0]]

            # Store tags in sample details (or a new field if we added one)
            # For now, we just check if it contradicts caption or is useful info
            # We add an INFO issue just to show the tags

            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Detected Scene Tags: {', '.join(detected_tags)}",
                    details={"tags": detected_tags, "probs": top_probs.tolist()},
                )
            )

        except Exception as e:
            logger.warning(f"Scene tagging failed: {e}")

        return sample

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            return load_representative_frame(sample.path, color="bgr")
        except Exception:
            return None

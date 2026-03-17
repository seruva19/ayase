import logging
import cv2
import numpy as np
from PIL import Image
import os
import json
from typing import Optional
from pathlib import Path
import urllib.request

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Original: https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth
AESTHETIC_MLP_URL = (
    "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/"
    "aesthetic_scoring/sac+logos+ava1-l14-linearMSE.pth"
)
AESTHETIC_MLP_FILENAME = "sac+logos+ava1-l14-linearMSE.pth"


class AestheticScoringModule(PipelineModule):
    name = "aesthetic_scoring"
    description = "Calculates aesthetic score (1-10) using LAION-Aesthetics MLP"
    default_config = {"models_dir": "models"}

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._mlp = None
        self._device = "cpu"
        self._ml_available = False

        # URL for the weights (not downloading here, just reference)
        # https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac+logos+ava1-l14-linearMSE.pth

    def on_mount(self) -> None:
        super().on_mount()
        try:
            import torch
            import torch.nn as nn
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Aesthetic Scoring models on {self._device}...")
            
            models_dir = self.config.get("models_dir", "models")

            # Load CLIP
            model_name = "openai/clip-vit-large-patch14"
            self._model = CLIPModel.from_pretrained(model_name, cache_dir=models_dir, use_safetensors=True).to(self._device)
            self._processor = CLIPProcessor.from_pretrained(model_name, cache_dir=models_dir)

            # Load MLP Head
            weight_dir = Path(models_dir) / "aesthetic_scoring"
            weight_dir.mkdir(parents=True, exist_ok=True)
            weight_path = weight_dir / AESTHETIC_MLP_FILENAME
            if not weight_path.exists():
                urllib.request.urlretrieve(AESTHETIC_MLP_URL, weight_path)
            
            # The official weight is a simple serialized torch dict or similar.
            # Usually it's just a linear layer weight (768, 1) and bias.
            
            state_dict = torch.load(str(weight_path), map_location=self._device, weights_only=True)
            
            # The model is a simple Linear(768, 1) usually, or MLP.
            # Inspection of the .pth usually reveals keys like 'weight', 'bias'.
            # LAION V2 is a simple linear regressor on normalized embeddings.
            
            self._linear = nn.Linear(768, 1)
            self._linear.weight.data = state_dict['weight'].to(self._device)
            self._linear.bias.data = state_dict['bias'].to(self._device)
            self._linear.to(self._device)
            
            self._ml_available = True
            logger.info("Loaded LAION-Aesthetics V2 predictor.")

        except ImportError:
            logger.warning("Transformers/Torch not installed. Aesthetic scoring disabled.")
        except Exception as e:
            logger.error(f"Failed to load Aesthetic models: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            from ayase.utils.sampling import FrameSampler
            frames = FrameSampler.sample_frames(sample.path, num_frames=8)
            
            if not frames:
                return sample

            import torch
            from PIL import Image

            # Batch process frames if possible to save overhead?
            # For simplicity and safety with varying sizes, we loop.
            
            scores = []
            
            for image in frames:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)

                inputs = self._processor(images=pil_image, return_tensors="pt").to(self._device)

                with torch.no_grad():
                    image_features = self._model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

                    # Predict score
                    score_tensor = self._linear(image_features.float())
                    scores.append(score_tensor.item())

            if not scores:
                return sample

            avg_score = sum(scores) / len(scores)

            # Check if we have quality_metrics
            if sample.quality_metrics is None:
                from ayase.models import QualityMetrics

                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.aesthetic_score = float(avg_score)

            if avg_score < 4.5:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Low aesthetic score: {avg_score:.2f}",
                        details={"score": avg_score, "frame_scores": scores},
                    )
                )

        except Exception as e:
            logger.warning(f"Aesthetic scoring failed: {e}")

        return sample

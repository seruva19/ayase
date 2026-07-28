"""Aesthetic scoring using the LAION-Aesthetics V2 MLP on CLIP-ViT-L/14 features.

Returns aesthetic_mlp_score on the original 1-10 scale. Scores below 4.5 are
flagged as low aesthetic quality. Processes 8 uniformly sampled frames."""

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
from ayase.compat import extract_features

logger = logging.getLogger(__name__)

# Original: https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth
AESTHETIC_MLP_URL = "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/aesthetic_scoring/sac+logos+ava1-l14-linearMSE.pth"
AESTHETIC_MLP_FILENAME = "sac+logos+ava1-l14-linearMSE.pth"


class AestheticScoringModule(PipelineModule):
    name = "aesthetic_scoring"
    description = "Calculates aesthetic score (1-10) using LAION-Aesthetics MLP"
    default_config = {"models_dir": "models"}
    metric_groups = {
        "aesthetic_mlp_score": "aesthetic",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._mlp = None
        self._device = "cpu"
        self._ml_available = False

        # URL for the weights (not downloading here, just reference)
        # https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac+logos+ava1-l14-linearMSE.pth

    def setup(self) -> None:
        try:
            import torch
            import torch.nn as nn
            from transformers import CLIPModel, CLIPProcessor
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            logger.info(f"Loading Aesthetic Scoring models on {self._device}...")
            
            models_dir = self.config.get("models_dir", "models")

            # Load CLIP
            model_name = "openai/clip-vit-large-patch14"
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

            # Load MLP Head
            weight_dir = Path(models_dir) / "aesthetic_scoring"
            weight_dir.mkdir(parents=True, exist_ok=True)
            weight_path = weight_dir / AESTHETIC_MLP_FILENAME
            if not weight_path.exists():
                urllib.request.urlretrieve(AESTHETIC_MLP_URL, weight_path)
            
            # The upstream weight is a simple serialized torch dict or similar.
            # Usually it's just a linear layer weight (768, 1) and bias.
            
            state_dict = torch.load(str(weight_path), map_location=self._device, weights_only=True)
            
            # LAION Aesthetic V2 is an MLP: 768 -> 1024 -> 128 -> 64 -> 16 -> 1
            # Keys in state_dict: layers.0.weight, layers.0.bias, ..., layers.7.weight, layers.7.bias
            class _AestheticMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(768, 1024), nn.Dropout(0.2),
                        nn.Linear(1024, 128), nn.Dropout(0.2),
                        nn.Linear(128, 64), nn.Dropout(0.1),
                        nn.Linear(64, 16),
                        nn.Linear(16, 1),
                    )
                def forward(self, x):
                    return self.layers(x)

            self._linear = _AestheticMLP()
            self._linear.load_state_dict(state_dict)
            self._linear.to(self._device)
            self._linear.eval()
            
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
            from ayase.image import sample_frames
            frames = sample_frames(sample.path, max_frames=8, color="rgb")

            if not frames:
                return sample

            import torch
            from PIL import Image

            # sample_frames returns RGB read-only views; copy for PIL.
            pil_images = [
                Image.fromarray(np.ascontiguousarray(image)) for image in frames
            ]
            inputs = self._processor(images=pil_images, return_tensors="pt").to(self._device)

            with torch.no_grad():
                image_features = extract_features(self._model.get_image_features(**inputs))
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                score_tensor = self._linear(image_features.float()).flatten()
                scores = [float(value) for value in score_tensor.detach().cpu().tolist()]

            if not scores:
                return sample

            avg_score = sum(scores) / len(scores)

            # Check if we have quality_metrics
            if sample.quality_metrics is None:
                from ayase.models import QualityMetrics

                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.aesthetic_mlp_score = float(avg_score)

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

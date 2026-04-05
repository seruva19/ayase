"""Watermark and AI-generated image detection using ResNet-50 or HuggingFace classifier.

Two-tier detection: custom ResNet-50 weights for watermarks, or HuggingFace
AI-image-detector as fallback. Returns watermark_probability or ai_generated_probability (0-1)."""

import logging

import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class WatermarkClassificationModule(PipelineModule):
    name = "watermark_classifier"
    description = "Classifies video for watermarks using a pretrained model or custom ResNet-50 weights"
    default_config = {
        "model_weights_path": "",  # Path to custom .pth watermark classifier (optional)
        "hf_model": "umm-maybe/AI-image-detector",  # HuggingFace fallback: AI-generated image detector
        "threshold": 0.5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_weights_path = self.config.get("model_weights_path", "")
        self.threshold = self.config.get("threshold", 0.5)

        self._model = None
        self._hf_pipe = None
        self._device = "cpu"
        self._ml_available = False
        self._use_hf = False
        self._transform = None

    def setup(self):
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            logger.warning("torch not installed. Watermark classifier disabled.")
            return

        # Path 1: User-provided custom weights (ResNet-50 binary classifier)
        if self.model_weights_path:
            try:
                import torch
                import torchvision.transforms as transforms
                from torchvision import models

                logger.info(f"Loading Watermark Classifier (ResNet-50) from {self.model_weights_path}...")
                self._model = models.resnet50(weights=None)
                num_ftrs = self._model.fc.in_features
                self._model.fc = torch.nn.Linear(num_ftrs, 1)
                state_dict = torch.load(
                    self.model_weights_path, map_location=self._device, weights_only=True
                )
                self._model.load_state_dict(state_dict)
                self._model.to(self._device)
                self._model.eval()
                self._transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                self._ml_available = True
                self._use_hf = False
                return
            except Exception as e:
                logger.warning(f"Failed to load custom weights: {e}. Trying HuggingFace fallback.")

        # Path 2: HuggingFace pipeline fallback
        # Note: umm-maybe/AI-image-detector detects AI-generated images, not watermarks.
        # The score is stored as ai_generated_probability when using this fallback.
        try:
            from transformers import pipeline as hf_pipeline
            hf_model = self.config.get("hf_model", "umm-maybe/AI-image-detector")
            logger.info(f"Loading AI-image detector from HuggingFace ({hf_model})...")
            self._hf_pipe = hf_pipeline("image-classification", model=hf_model, device=self._device)
            self._ml_available = True
            self._use_hf = True
        except Exception as e:
            logger.warning(f"Failed to load image classifier: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        image = self._load_image(sample)
        if image is None:
            return sample

        try:
            import torch

            if self._use_hf:
                from PIL import Image
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                results = self._hf_pipe(pil_image)
                # HF fallback detects AI-generated images, not watermarks
                score = 0.0
                for r in results:
                    label = r["label"].lower()
                    if any(kw in label for kw in ("artificial", "ai", "watermark", "fake")):
                        score = max(score, r["score"])
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                input_tensor = self._transform(image).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    output = self._model(input_tensor)
                    score = torch.sigmoid(output).item()

            if sample.quality_metrics is None:
                from ayase.models import QualityMetrics
                sample.quality_metrics = QualityMetrics()

            if self._use_hf:
                # HF model detects AI-generated images, not watermarks
                sample.quality_metrics.ai_generated_probability = float(score)
            else:
                sample.quality_metrics.watermark_probability = float(score)

            if score > self.threshold:
                label = "AI-Generated Image" if self._use_hf else "Watermark/Artifact"
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"{label} Detected (Score: {score:.2f})",
                        details={"score": score, "detector": "ai_image" if self._use_hf else "watermark"},
                        recommendation="Review image origin." if self._use_hf else "Discard video or perform watermark removal."
                    )
                )

        except Exception as e:
            logger.warning(f"Watermark classification failed: {e}")

        return sample

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Check middle frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
                ret, frame = cap.read()
                cap.release()
                return frame if ret else None
            else:
                return cv2.imread(str(sample.path))
        except Exception:
            return None

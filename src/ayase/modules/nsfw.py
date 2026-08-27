"""NSFW content detection using a ViT-based image classification model.

Classifies frames as normal or NSFW using Falconsai/nsfw_image_detection.
Returns the peak NSFW probability and, for video, the fraction of sampled time
whose probability crosses the configured risk threshold. Higher is less safe."""

import logging
from PIL import Image
from typing import Optional

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule
from ayase.utils.sampling import FrameSampler

logger = logging.getLogger(__name__)

class NSFWModule(PipelineModule):
    name = "nsfw"
    description = "Detects NSFW (adult/violent) content using ViT"
    default_config = {
        "model_name": "Falconsai/nsfw_image_detection",
        "model_revision": "04367978d3474804ab1a00a9bd6548b741764069",
        "threshold": 0.5,
        "num_frames": 8,
    }
    models = [
        {
            "id": "Falconsai/nsfw_image_detection",
            "type": "huggingface",
            "revision": "04367978d3474804ab1a00a9bd6548b741764069",
            "task": "Per-frame NSFW classification",
            "auto_download": True,
        }
    ]
    metric_info = {
        "temporal_risk_rate": (
            "Fraction of uniformly sampled video frames at or above the NSFW "
            "threshold (0-1, higher=less safe)"
        )
    }
    metric_groups = {
        "nsfw_score": "safety",
        "temporal_risk_rate": "safety",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "Falconsai/nsfw_image_detection")
        configured_revision = self.config.get("model_revision")
        if self.model_name != "Falconsai/nsfw_image_detection" and not (
            config and "model_revision" in config
        ):
            configured_revision = None
        self.model_revision = configured_revision
        self.threshold = self.config.get("threshold", 0.5)
        self.num_frames = self.config.get("num_frames", 8)
        
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        try:
            from transformers import AutoModelForImageClassification, ViTImageProcessor

            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            logger.info(f"Loading NSFW model ({self.model_name}) on {self._device}...")

            models_dir = self.config.get("models_dir", "models")

            self._processor = ViTImageProcessor.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                cache_dir=models_dir,
            )
            self._model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                cache_dir=models_dir,
            ).to(self._device).eval()

            self._ml_available = True
            self._backend = "transformers"
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup NSFW module: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            import torch
            frames = FrameSampler.sample_frames(sample.path, num_frames=self.num_frames)
            if not frames:
                return sample

            import cv2
            pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
            
            inputs = self._processor(images=pil_images, return_tensors="pt").to(self._device)
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = logits.softmax(dim=-1)
                
            # Labels: usually ['normal', 'nsfw']
            # We need to check the exact mapping
            id2label = self._model.config.id2label
            nsfw_idx = -1
            for idx, label in id2label.items():
                if "nsfw" in label.lower():
                    nsfw_idx = idx
                    break
            
            if nsfw_idx == -1:
                logger.warning("Could not find NSFW label in model config. Skipping.")
                return sample

            nsfw_probs = probs[:, nsfw_idx].cpu().tolist()
            max_nsfw = max(nsfw_probs)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.nsfw_score = float(max_nsfw)
            if sample.is_video:
                sample.quality_metrics.temporal_risk_rate = self._temporal_risk_rate(
                    nsfw_probs, self.threshold
                )

            if max_nsfw > self.threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"NSFW Content Detected! (Max Prob: {max_nsfw:.2f})",
                        details={"nsfw_scores": nsfw_probs, "model": self.model_name},
                    )
                )

        except Exception as e:
            logger.warning(f"NSFW processing failed for {sample.path}: {e}")

        return sample

    @staticmethod
    def _temporal_risk_rate(probabilities, threshold: float) -> float:
        """Return the sampled-time occupancy of thresholded risk."""
        if not probabilities:
            return 0.0
        risky = sum(float(probability) >= threshold for probability in probabilities)
        return float(risky / len(probabilities))

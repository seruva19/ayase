"""BLIP image-text matching score for prompt alignment.

Uses BLIP-ITM when available to score whether an image or sampled video frames
match the associated caption. Scores are normalized to 0-1, higher is better.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class BLIPScoreModule(PipelineModule):
    name = "blip_score"
    description = "BLIP image-text matching alignment score"
    default_config = {
        "model_name": "Salesforce/blip-itm-large-coco",
        "max_frames": 8,
        "warning_threshold": 0.4,
        "device": "auto",
    }
    models = [
        {
            "id": "Salesforce/blip-itm-large-coco",
            "type": "huggingface",
            "task": "BLIP image-text matching",
        },
    ]
    metric_info = {
        "blip_score": "BLIP image-text matching probability (0-1, higher=better)",
    }
    metric_groups = {
        "blip_score": "alignment",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "Salesforce/blip-itm-large-coco")
        self.max_frames = self.config.get("max_frames", 8)
        self.warning_threshold = self.config.get("warning_threshold", 0.4)
        self.device_config = self.config.get("device", "auto")
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self) -> None:
        try:
            import torch
            from transformers import BlipForImageTextRetrieval, BlipProcessor

            from ayase.config import resolve_model_path

            if self.device_config == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.device_config

            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.model_name, models_dir)
            self._processor = BlipProcessor.from_pretrained(resolved)
            self._model = BlipForImageTextRetrieval.from_pretrained(resolved).to(self._device)
            self._model.eval()
            self._ml_available = True
            logger.info("BLIP score initialised with %s on %s", self.model_name, self._device)
        except ImportError:
            logger.warning("BLIP score requires torch and transformers")
        except Exception as e:
            logger.warning("Failed to setup BLIP score: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        caption = self._caption_text(sample)
        if not caption:
            return sample

        try:
            frames = arrays_to_pil(sample_frames(sample.path, max_frames=self.max_frames, color="rgb"))
            if not frames:
                return sample

            scores = [score for image in frames if (score := self._score_image(image, caption)) is not None]
            if not scores:
                return sample
            score = float(np.mean(scores))

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.blip_score = score

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low BLIP alignment: {score:.3f}",
                        details={"blip_score": score, "caption": caption[:80]},
                    )
                )
        except Exception as e:
            logger.warning("BLIP score failed for %s: %s", sample.path, e)
        return sample

    def _caption_text(self, sample: Sample) -> Optional[str]:
        if sample.caption and sample.caption.text:
            return sample.caption.text
        sidecar = sample.path.with_suffix(".txt")
        try:
            if sidecar.exists():
                return sidecar.read_text(encoding="utf-8").strip()
        except Exception:
            logger.debug("Failed to read caption sidecar %s", sidecar)
        return None

    def _score_image(self, image, caption: str) -> Optional[float]:
        try:
            import torch

            inputs = self._processor(images=image, text=caption, return_tensors="pt").to(self._device)
            with torch.no_grad():
                try:
                    outputs = self._model(**inputs, use_itm_head=True)
                except TypeError:
                    outputs = self._model(**inputs)

            logits = getattr(outputs, "itm_score", None)
            if logits is None:
                logits = getattr(outputs, "logits", None)
            if logits is None:
                logits = outputs[0] if isinstance(outputs, (tuple, list)) and outputs else None
            if logits is None:
                return None

            logits = logits.detach()
            if logits.ndim == 2 and logits.shape[-1] >= 2:
                return float(torch.softmax(logits, dim=-1)[0, -1].item())
            return float(torch.sigmoid(logits.reshape(-1)[0]).item())
        except Exception as e:
            logger.debug("BLIP frame scoring failed: %s", e)
            return None

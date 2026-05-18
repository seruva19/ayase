"""Subject consistency via DINOv2-base pairwise frame similarity (VBench-style).

Computes all-pairs cosine similarity of CLS-token embeddings across frames.
Returns subject_consistency (0-1, higher = more consistent). Warns below 0.6."""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List

from ayase.image import sample_frames
from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SubjectConsistencyModule(PipelineModule):
    name = "subject_consistency"
    description = "Subject consistency using DINOv2-base (all pairwise frame similarity)"

    default_config = {
        "model_name": "facebook/dinov2-base",
        "max_frames": 16,
        "warning_threshold": 0.6,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_frames = self.config.get("max_frames", 16)
        self.warning_threshold = self.config.get("warning_threshold", 0.6)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            model_name = self.config.get("model_name", "facebook/dinov2-base")
            models_dir = self.config.get("models_dir", "models")
            logger.info(f"Loading {model_name} on {self._device}...")

            def load_dino():
                processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=models_dir)
                model = from_pretrained_with_attention(
                    AutoModel,
                    model_name,
                    self.config,
                    device=self._device,
                    cache_dir=models_dir,
                    use_safetensors=True,
                ).to(self._device).eval()
                return model, processor

            self._model, self._processor = shared_runtime_resource(
                self,
                (
                    "hf_vision",
                    model_name,
                    self._device,
                    str(self.config.get("attention_backend", "auto")),
                    "safetensors",
                ),
                load_dino,
            )
            self._ml_available = True

        except ImportError:
            logger.warning("Transformers/Torch not installed. DINO checks disabled.")
        except Exception as e:
            logger.error(f"Failed to load DINOv2: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            frames = self._load_frames(sample)
            if len(frames) < 2:
                return sample

            import torch
            import torch.nn.functional as F

            pil_images = [
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames
            ]
            inputs = self._processor(images=pil_images, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
                embeddings = F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=-1)

            # All pairwise cosine similarities (VBench style)
            sim_matrix = embeddings @ embeddings.T
            pair_indices = torch.triu_indices(
                embeddings.size(0),
                embeddings.size(0),
                offset=1,
                device=sim_matrix.device,
            )
            avg_consistency = float(sim_matrix[pair_indices[0], pair_indices[1]].mean().item())

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.subject_consistency = avg_consistency

            if avg_consistency < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low subject consistency: {avg_consistency:.2f}",
                        details={"consistency_score": avg_consistency},
                    )
                )

        except Exception as e:
            logger.warning(f"Subject consistency check failed: {e}")

        return sample

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            frames = sample_frames(sample.path, max_frames=self.max_frames, color="bgr")
        except Exception as e:
            logger.debug(f"Failed to load frames for subject consistency: {e}")
            frames = []
        return frames


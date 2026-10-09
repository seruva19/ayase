"""VBench Subject Consistency over DINO ViT-B/16 CLS embeddings.

Per frame i>=1 the score is ``(max(0, cos(f_i, f_{i-1})) + max(0, cos(f_i,
f_0))) / 2``; the reported value is the mean over all frames (VBench
protocol — every frame is embedded, no sampling). Returns
``subject_consistency`` (0-1, higher = more consistent). Warns below 0.6."""

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
    provenance = "adapted"
    sources = {
        "subject_consistency": "VBench subject consistency (Huang et al. CVPR 2024) — https://github.com/Vchitect/VBench",
    }
    deviations = {
        "subject_consistency": "Preprocessor is HF AutoImageProcessor (resize shortest edge 256 + center crop 224); upstream dino_transform resizes the shortest edge to 224 without cropping — embeddings differ at the ~0.003 level",
    }
    description = "Subject consistency using DINO ViT-B/16 (VBench per-frame protocol)"

    default_config = {
        "model_name": "facebook/dino-vitb16",
        "revision": None,
        "max_frames": 0,
        "warning_threshold": 0.6,
    }
    metric_groups = {
        "subject_consistency": "temporal",
    }

    def __init__(self, config=None):
        super().__init__(config)
        # VBench embeds every frame; max_frames <= 0 keeps that, a positive
        # value is an explicit non-default sampling cap.
        self.max_frames = self.config.get("max_frames", 0)
        self.warning_threshold = self.config.get("warning_threshold", 0.6)
        self.revision = self.config.get("revision")
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
            model_name = self.config.get("model_name", "facebook/dino-vitb16")
            models_dir = self.config.get("models_dir", "models")
            revision_kwargs = self._revision_kwargs()
            logger.info(f"Loading {model_name} on {self._device}...")

            def load_dino():
                processor = AutoImageProcessor.from_pretrained(
                    model_name, cache_dir=models_dir, **revision_kwargs
                )
                model = from_pretrained_with_attention(
                    AutoModel,
                    model_name,
                    self.config,
                    device=self._device,
                    cache_dir=models_dir,
                    use_safetensors=True,
                    **revision_kwargs,
                ).to(self._device).eval()
                return model, processor

            resource_key = (
                "hf_vision",
                model_name,
                self._device,
                str(self.config.get("attention_backend", "auto")),
                "safetensors",
            )
            if revision_kwargs:
                resource_key += ("revision", revision_kwargs["revision"])
            self._model, self._processor = shared_runtime_resource(
                self,
                resource_key,
                load_dino,
            )
            self._ml_available = True

        except ImportError:
            logger.warning("Transformers/Torch not installed. DINO checks disabled.")
        except Exception as e:
            logger.error(f"Failed to load DINO ViT-B/16: {e}")

    def _revision_kwargs(self) -> dict:
        """Return a validated optional Hugging Face revision argument."""
        if self.revision is None:
            return {}
        if not isinstance(self.revision, str):
            raise ValueError("revision must be a string or None")
        revision = self.revision.strip()
        if not revision or len(revision) > 256:
            raise ValueError("revision must be a non-empty string of at most 256 characters")
        return {"revision": revision}

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

            # VBench: for each frame i>=1, (max(0, sim to previous) +
            # max(0, sim to first)) / 2; the metric is the mean over frames.
            first = embeddings[0]
            prev = embeddings[0]
            frame_sims = []
            for i in range(1, embeddings.size(0)):
                cur = embeddings[i]
                sim_pre = max(0.0, (prev @ cur).item())
                sim_fir = max(0.0, (first @ cur).item())
                frame_sims.append((sim_pre + sim_fir) / 2)
                prev = cur
            if not frame_sims:
                return sample
            avg_consistency = float(np.mean(frame_sims))

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
            limit = self.max_frames if self.max_frames > 0 else 1_000_000
            frames = sample_frames(sample.path, max_frames=limit, color="bgr")
        except Exception as e:
            logger.debug(f"Failed to load frames for subject consistency: {e}")
            frames = []
        return frames


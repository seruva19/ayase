"""Background consistency via CLIP pairwise frame similarity (VBench-style).

Computes all-pairs cosine similarity across uniformly sampled frames.
Returns background_consistency (0-1, higher = more consistent). Warns below 0.5."""

import logging
import numpy as np
from typing import List

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_clip_image_feature_groups,
    cached_clip_image_features,
    media_state_key,
)

logger = logging.getLogger(__name__)


class BackgroundConsistencyModule(PipelineModule):
    name = "background_consistency"
    description = "Background consistency using CLIP (all pairwise frame similarity)"

    default_config = {
        "model_name": "openai/clip-vit-base-patch32",
        "max_frames": 16,
        "warning_threshold": 0.5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_frames = self.config.get("max_frames", 16)
        self.warning_threshold = self.config.get("warning_threshold", 0.5)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            model_name = self.config.get("model_name", "openai/clip-vit-base-patch32")
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(model_name, models_dir)
            logger.info(f"Loading CLIP for Background Consistency on {self._device}...")

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=self._device,
                    use_safetensors=True,
                ).to(self._device).eval()
                processor = CLIPProcessor.from_pretrained(resolved)
                return model, processor

            self._model, self._processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved,
                    self._device,
                    str(self.config.get("attention_backend", "auto")),
                    "safetensors",
                ),
                load_clip,
            )
            self._ml_available = True

        except ImportError:
            logger.warning("Transformers/Torch not installed. Background Consistency disabled.")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            frames = self._load_frames(sample)
            if len(frames) < 2:
                return sample

            pil_frames = arrays_to_pil(frames)
            model_name = self.config.get("model_name", "openai/clip-vit-base-patch32")
            embeddings = cached_clip_image_features(
                self,
                self._model,
                self._processor,
                pil_frames,
                model_key=model_name,
                device=self._device,
                cache_key=(self.max_frames, media_state_key(sample.path)),
            )

            self._apply_score(sample, embeddings)

        except Exception as e:
            logger.warning(f"Background consistency check failed: {e}")

        return sample

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if not self._ml_available:
            return samples

        try:
            prepared = []
            image_groups = []
            cache_keys = []
            for sample in samples:
                if not sample.is_video:
                    continue
                frames = self._load_frames(sample)
                if len(frames) < 2:
                    continue
                prepared.append(sample)
                image_groups.append(arrays_to_pil(frames))
                cache_keys.append((self.max_frames, media_state_key(sample.path)))

            if not prepared:
                return samples

            model_name = self.config.get("model_name", "openai/clip-vit-base-patch32")
            feature_groups = cached_clip_image_feature_groups(
                self,
                self._model,
                self._processor,
                image_groups,
                model_key=model_name,
                device=self._device,
                cache_keys=cache_keys,
            )
            for sample, embeddings in zip(prepared, feature_groups):
                self._apply_score(sample, embeddings)
        except Exception as e:
            logger.warning("Background consistency batch check failed: %s", e)

        return samples

    def _apply_score(self, sample: Sample, embeddings) -> None:
        if embeddings is None or embeddings.size(0) < 2:
            return

        import torch.nn.functional as F

        # All pairwise cosine similarities (VBench style)
        similarities = []
        for i in range(embeddings.size(0)):
            for j in range(i + 1, embeddings.size(0)):
                # embeddings is a 2D [n_frames, D] tensor from cached_clip_image_features;
                # indexing yields a 1D vector, so default dim=1 in cosine_similarity errors.
                sim = F.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1], dim=1).item()
                similarities.append(sim)

        avg_consistency = float(np.mean(similarities))

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.background_consistency = avg_consistency

        if avg_consistency < self.warning_threshold:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low background consistency: {avg_consistency:.2f} (Scene might have changed)",
                    details={"consistency_score": avg_consistency},
                )
            )

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            return sample_frames(sample.path, max_frames=self.max_frames, color="rgb")
        except Exception as e:
            logger.debug(f"Failed to load frames for background consistency: {e}")
        return []





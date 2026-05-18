"""VQAThinker — Generalizable Explainable VQA via RL (2025).

RL-based explainable video quality assessment that uses quality
reasoning via CLIP backbone. Computes score with rationale using
softmax over quality-level descriptions with temperature scaling.

Implementation: CLIP backbone for quality reasoning. Extract
quality-relevant features and compute score via zero-shot
classification over quality level descriptions.

vqathinker_score — higher = better (0-1)
"""

import logging
from typing import Optional, List

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_clip_image_feature_groups,
    cached_clip_image_features,
    cached_clip_text_features,
    media_state_key,
)

logger = logging.getLogger(__name__)

# Quality level descriptions for CLIP zero-shot scoring
# Inspired by the VQAThinker paper's reasoning approach
_QUALITY_LEVELS = [
    "an extremely low quality video with severe distortions and artifacts",
    "a very low quality video with significant noise and blur",
    "a low quality video with noticeable compression artifacts",
    "a below average quality video with some visible imperfections",
    "a fair quality video with minor issues",
    "an acceptable quality video with mostly clean visuals",
    "a good quality video with clear details",
    "a high quality video with sharp and clean visuals",
    "a very high quality video with excellent detail preservation",
    "an outstanding quality video with perfect visual fidelity",
]

# Numeric quality values mapped to each level (0-1 scale)
_QUALITY_VALUES = np.linspace(0.05, 0.95, len(_QUALITY_LEVELS))


class VQAThinkerModule(PipelineModule):
    name = "vqathinker"
    description = "VQAThinker RL-based explainable VQA (2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
        "temperature": 0.07,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.temperature = self.config.get("temperature", 0.07)
        self._backend = None
        self._ml_available = False
        self._device = "cpu"

        self._clip_model = None
        self._clip_processor = None
        self._quality_text_embeds = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            from transformers import CLIPModel, CLIPProcessor
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))

            clip_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(clip_name, models_dir)

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=self._device,
                ).to(self._device).eval()
                processor = CLIPProcessor.from_pretrained(resolved)
                return model, processor

            self._clip_model, self._clip_processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved,
                    self._device,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load_clip,
            )

            # Pre-compute text embeddings for quality levels
            self._quality_text_embeds = cached_clip_text_features(
                self,
                self._clip_model,
                self._clip_processor,
                _QUALITY_LEVELS,
                model_key=clip_name,
                device=self._device,
                truncation=True,
            )  # [10, D]

            self._ml_available = True
            self._backend = "clip_thinker"
            logger.info(
                "VQAThinker (CLIP quality reasoning) initialised on %s",
                self._device,
            )

        except Exception as e:
            logger.warning("VQAThinker setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_score(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.vqathinker_score = score
        except Exception as e:
            logger.warning("VQAThinker failed for %s: %s", sample.path, e)
        return sample

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if not self._ml_available:
            return samples

        try:
            prepared = []
            image_groups = []
            cache_keys = []
            for sample in samples:
                frames = self._extract_frames(sample)
                if not frames:
                    continue
                prepared.append(sample)
                image_groups.append(arrays_to_pil(frames))
                cache_keys.append((self.subsample, media_state_key(sample.path)))

            if not prepared:
                return samples

            feature_groups = cached_clip_image_feature_groups(
                self,
                self._clip_model,
                self._clip_processor,
                image_groups,
                model_key=self.config.get("clip_model", "openai/clip-vit-base-patch32"),
                device=self._device,
                cache_keys=cache_keys,
            )
            for sample, image_features in zip(prepared, feature_groups):
                score = self._score_image_features(image_features)
                if score is None:
                    continue
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.vqathinker_score = score
        except Exception as e:
            logger.warning("VQAThinker batch failed: %s", e)

        return samples

    def _compute_score(self, sample: Sample) -> Optional[float]:
        """CLIP quality reasoning with temperature-scaled softmax."""

        frames = self._extract_frames(sample)
        if not frames:
            return None
        image_features = cached_clip_image_features(
            self,
            self._clip_model,
            self._clip_processor,
            arrays_to_pil(frames),
            model_key=self.config.get("clip_model", "openai/clip-vit-base-patch32"),
            device=self._device,
            cache_key=(self.subsample, media_state_key(sample.path)),
        )
        return self._score_image_features(image_features)

    def _score_image_features(self, image_features) -> Optional[float]:
        """CLIP quality reasoning with temperature-scaled softmax."""
        import torch

        if image_features is None or image_features.size(0) == 0:
            return None

        frame_scores = []

        with torch.no_grad():
            for img_feats in image_features:
                img_feats = img_feats.unsqueeze(0)

                # Compute similarity to each quality level
                similarities = (img_feats @ self._quality_text_embeds.T).squeeze(0)  # [10]

                # Temperature-scaled softmax
                logits = similarities / self.temperature
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                # Expected quality value (weighted sum)
                score = float(np.sum(probs * _QUALITY_VALUES))
                frame_scores.append(score)

        if not frame_scores:
            return None

        return float(np.clip(np.mean(frame_scores), 0.0, 1.0))

    def _extract_frames(self, sample: Sample):
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("VQAThinker frame loading failed for %s: %s", sample.path, e)
            return []

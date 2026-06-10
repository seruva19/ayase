"""UGVQ — Unified Generated Video Quality (ACM TOMM 2024).

Multi-dimensional quality assessment for generated video content.
Evaluates visual fidelity, temporal consistency, and content
naturalness using CLIP embeddings.

Implementation: CLIP for multi-dimensional quality scoring via
quality-aware text prompts, frame embedding stability for temporal
consistency, and real-vs-generated discrimination.

ugvq_score — higher = better (0-1)
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

# Quality assessment prompt sets for CLIP zero-shot scoring
_FIDELITY_PROMPTS = {
    "high": [
        "a high quality photo with sharp details",
        "a clear and well-defined image",
        "a photo with excellent visual quality",
        "a crisp and detailed image",
    ],
    "low": [
        "a blurry low quality photo",
        "a noisy and distorted image",
        "a photo with poor visual quality",
        "an image with visible artifacts",
    ],
}

_NATURALNESS_PROMPTS = {
    "natural": [
        "a natural realistic photograph",
        "a photo of a real scene",
        "a natural looking image",
    ],
    "generated": [
        "an AI generated image",
        "a synthetic computer generated image",
        "an artificial image with unnatural features",
    ],
}


class UGVQModule(PipelineModule):
    name = "ugvq"
    description = "UGVQ unified generated video quality (TOMM 2024)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }
    metric_groups = {
        "ugvq_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = None
        self._ml_available = False
        self._device = "cpu"

        self._clip_model = None
        self._clip_processor = None
        self._fidelity_high_embeds = None
        self._fidelity_low_embeds = None
        self._natural_embeds = None
        self._generated_embeds = None

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

            # Pre-compute text embeddings for quality prompts
            self._fidelity_high_embeds = self._encode_texts(
                _FIDELITY_PROMPTS["high"]
            )
            self._fidelity_low_embeds = self._encode_texts(
                _FIDELITY_PROMPTS["low"]
            )
            self._natural_embeds = self._encode_texts(
                _NATURALNESS_PROMPTS["natural"]
            )
            self._generated_embeds = self._encode_texts(
                _NATURALNESS_PROMPTS["generated"]
            )

            self._ml_available = True
            self._backend = "clip_ugvq"
            logger.info("UGVQ (CLIP multi-dim quality) initialised on %s", self._device)

        except Exception as e:
            logger.warning("UGVQ setup failed: %s", e)

    def _encode_texts(self, texts: List[str]):
        """Encode text prompts into normalized CLIP embeddings."""
        return cached_clip_text_features(
            self,
            self._clip_model,
            self._clip_processor,
            texts,
            model_key=self.config.get("clip_model", "openai/clip-vit-base-patch32"),
            device=self._device,
        )

    def _encode_images(self, sample: Sample, frames: List[np.ndarray]):
        """Encode RGB frames into normalized CLIP embeddings."""
        return cached_clip_image_features(
            self,
            self._clip_model,
            self._clip_processor,
            arrays_to_pil(frames),
            model_key=self.config.get("clip_model", "openai/clip-vit-base-patch32"),
            device=self._device,
            cache_key=(self.subsample, media_state_key(sample.path)),
        )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_score(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.ugvq_score = score
        except Exception as e:
            logger.warning("UGVQ failed for %s: %s", sample.path, e)
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
                sample.quality_metrics.ugvq_score = score
        except Exception as e:
            logger.warning("UGVQ batch failed for: %s", e)

        return samples

    def _compute_score(self, sample: Sample) -> Optional[float]:
        """Multi-dimensional CLIP-based quality assessment."""

        frames = self._extract_frames(sample)
        if not frames:
            return None
        return self._score_image_features(self._encode_images(sample, frames))

    def _score_image_features(self, image_features) -> Optional[float]:
        """Multi-dimensional CLIP-based quality assessment."""
        import torch

        if image_features is None or image_features.size(0) == 0:
            return None

        frame_embeddings = []
        fidelity_scores = []
        naturalness_scores = []

        with torch.no_grad():
            for img_embed in image_features:
                img_embed = img_embed.unsqueeze(0)  # [1, D]
                frame_embeddings.append(img_embed)

                # Dimension 1: Visual Fidelity
                # Softmax over quality-positive vs quality-negative prompts
                high_sim = (img_embed @ self._fidelity_high_embeds.T).mean().item()
                low_sim = (img_embed @ self._fidelity_low_embeds.T).mean().item()
                fidelity = 1.0 / (1.0 + np.exp(-(high_sim - low_sim) * 10.0))
                fidelity_scores.append(fidelity)

                # Dimension 2: Content Naturalness
                nat_sim = (img_embed @ self._natural_embeds.T).mean().item()
                gen_sim = (img_embed @ self._generated_embeds.T).mean().item()
                naturalness = 1.0 / (1.0 + np.exp(-(nat_sim - gen_sim) * 10.0))
                naturalness_scores.append(naturalness)

        # Dimension 3: Temporal Consistency (frame embedding stability)
        temporal_score = 1.0
        if len(frame_embeddings) > 1:
            embeddings = torch.cat(frame_embeddings, dim=0)  # [T, D]
            consec_sims = []
            for i in range(embeddings.size(0) - 1):
                sim = (embeddings[i] @ embeddings[i + 1]).item()
                consec_sims.append(sim)
            # Map cosine similarity to quality score
            mean_sim = np.mean(consec_sims)
            temporal_score = float(np.clip((mean_sim - 0.5) * 2.0, 0.0, 1.0))

        # Unified score: weighted combination of dimensions
        avg_fidelity = float(np.mean(fidelity_scores))
        avg_naturalness = float(np.mean(naturalness_scores))

        score = (
            0.40 * avg_fidelity
            + 0.30 * temporal_score
            + 0.30 * avg_naturalness
        )

        return float(np.clip(score, 0.0, 1.0))

    def _extract_frames(self, sample: Sample):
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("UGVQ frame loading failed for %s: %s", sample.path, e)
            return []

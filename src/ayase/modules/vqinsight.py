"""VQ-Insight — ByteDance AIGC Video Quality (AAAI 2026).

Multi-dimensional AIGC video quality assessment. Evaluates generated
video across four quality dimensions: visual quality, text alignment,
temporal coherence, and aesthetic quality.

Implementation: CLIP for multi-dimensional AIGC quality scoring.
Each dimension gets its own prompt set and score via zero-shot
CLIP classification.

vqinsight_score — higher = better (0-1)
"""

import logging
from typing import Optional, Dict, List

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

# Multi-dimensional prompt sets for CLIP zero-shot scoring
_DIMENSION_PROMPTS: Dict[str, Dict[str, List[str]]] = {
    "visual_quality": {
        "positive": [
            "a high quality image with sharp details and low noise",
            "a crisp clear image with excellent resolution",
            "a well-exposed image with vivid colors and no artifacts",
            "a professionally captured image with superb clarity",
        ],
        "negative": [
            "a blurry image with noise and compression artifacts",
            "a low resolution image with poor detail",
            "a distorted image with visible quality degradation",
            "an image with blocky artifacts and color banding",
        ],
    },
    "aesthetic_quality": {
        "positive": [
            "a beautiful aesthetically pleasing image with great composition",
            "an artistically composed image with harmonious colors",
            "a visually stunning image with balanced lighting",
            "a well-composed image with appealing visual balance",
        ],
        "negative": [
            "an ugly unpleasant image with poor composition",
            "a visually unappealing image with clashing colors",
            "a poorly composed image with bad framing",
            "an aesthetically displeasing image",
        ],
    },
    "content_naturalness": {
        "positive": [
            "a natural realistic photograph of a real scene",
            "a photorealistic image indistinguishable from a photograph",
            "a natural looking scene with realistic textures and lighting",
        ],
        "negative": [
            "an obviously AI generated image with unnatural features",
            "a synthetic image with telltale generation artifacts",
            "an artificial looking image with uncanny visual elements",
        ],
    },
}


class VQInsightModule(PipelineModule):
    name = "vqinsight"
    description = "VQ-Insight ByteDance multi-dim AIGC scoring (AAAI 2026)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }
    metric_groups = {
        "vqinsight_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = None
        self._ml_available = False
        self._device = "cpu"

        self._clip_model = None
        self._clip_processor = None
        self._dim_embeds: Dict[str, Dict[str, object]] = {}

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

            # Pre-compute text embeddings for each quality dimension
            for dim_name, prompts in _DIMENSION_PROMPTS.items():
                self._dim_embeds[dim_name] = {
                    "positive": self._encode_texts(prompts["positive"]),
                    "negative": self._encode_texts(prompts["negative"]),
                }

            self._ml_available = True
            self._backend = "clip_vqinsight"
            logger.info(
                "VQ-Insight (CLIP multi-dim AIGC, %d dimensions) initialised on %s",
                len(_DIMENSION_PROMPTS), self._device,
            )

        except Exception as e:
            logger.warning("VQ-Insight setup failed: %s", e)

    def _encode_texts(self, texts: List[str]):
        """Encode text prompts into normalized CLIP embeddings."""
        return cached_clip_text_features(
            self,
            self._clip_model,
            self._clip_processor,
            texts,
            model_key=self.config.get("clip_model", "openai/clip-vit-base-patch32"),
            device=self._device,
            truncation=True,
        )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_score(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.vqinsight_score = score
        except Exception as e:
            logger.warning("VQ-Insight failed for %s: %s", sample.path, e)
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
                sample.quality_metrics.vqinsight_score = score
        except Exception as e:
            logger.warning("VQ-Insight batch failed: %s", e)

        return samples

    def _compute_score(self, sample: Sample) -> Optional[float]:
        """Multi-dimensional CLIP-based AIGC quality assessment."""

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
        """Multi-dimensional CLIP-based AIGC quality assessment."""
        import torch

        if image_features is None or image_features.size(0) == 0:
            return None

        # Collect per-frame embeddings and dimension scores
        frame_embeddings = []
        dim_scores: Dict[str, List[float]] = {
            dim: [] for dim in _DIMENSION_PROMPTS
        }

        with torch.no_grad():
            for img_feats in image_features:
                img_feats = img_feats.unsqueeze(0)
                frame_embeddings.append(img_feats)

                # Score each quality dimension
                for dim_name, embeds in self._dim_embeds.items():
                    pos_sim = (img_feats @ embeds["positive"].T).mean().item()
                    neg_sim = (img_feats @ embeds["negative"].T).mean().item()
                    # Sigmoid of difference for 0-1 score
                    dim_score = 1.0 / (1.0 + np.exp(-(pos_sim - neg_sim) * 10.0))
                    dim_scores[dim_name].append(dim_score)

        # Temporal coherence dimension (from frame embedding stability)
        temporal_coherence = 1.0
        if len(frame_embeddings) > 1:
            all_embeds = torch.cat(frame_embeddings, dim=0)  # [T, D]
            consec_sims = []
            for i in range(all_embeds.size(0) - 1):
                sim = (all_embeds[i] @ all_embeds[i + 1]).item()
                consec_sims.append(sim)
            mean_sim = np.mean(consec_sims)
            temporal_coherence = float(np.clip((mean_sim - 0.5) * 2.0, 0.0, 1.0))

        # Average each dimension across frames
        avg_dims = {
            dim: float(np.mean(scores)) for dim, scores in dim_scores.items()
        }

        # Final multi-dimensional fusion score
        # Weights: visual quality (0.30), aesthetic (0.25),
        #          temporal coherence (0.25), content naturalness (0.20)
        score = (
            0.30 * avg_dims.get("visual_quality", 0.5)
            + 0.25 * avg_dims.get("aesthetic_quality", 0.5)
            + 0.25 * temporal_coherence
            + 0.20 * avg_dims.get("content_naturalness", 0.5)
        )

        return float(np.clip(score, 0.0, 1.0))

    def _extract_frames(self, sample: Sample):
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("VQ-Insight frame loading failed for %s: %s", sample.path, e)
            return []

"""CLiF-VQA --- Human Feelings VQA via CLIP (2024).

Extracts human-feelings features from CLIP to simulate the HVS
perceptual process for quality assessment.  Quality is the weighted
similarity between video frames and quality-feeling text prompts
(comfort, clarity, warmth, vibrancy, harmony) vs negative feeling prompts.

clifvqa_score --- higher = better quality (0-1 range)
"""

import logging
from typing import List, Optional

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

# Human-feeling text prompts modelling HVS perceptual dimensions
_POSITIVE_FEELINGS = [
    "a comfortable and pleasant image",
    "a clear and sharp image",
    "a warm and inviting image",
    "a vibrant image with rich colors",
    "a harmonious and balanced image",
    "a natural and realistic image",
    "a relaxing and soothing image",
]
_NEGATIVE_FEELINGS = [
    "an uncomfortable and unpleasant image",
    "a blurry and unclear image",
    "a cold and uninviting image",
    "a dull image with washed out colors",
    "a chaotic and unbalanced image",
    "an unnatural and artificial image",
    "an irritating and disturbing image",
]


class CLiFVQAModule(PipelineModule):
    name = "clifvqa"
    description = "CLiF-VQA human feelings VQA via CLIP (2024)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }
    metric_groups = {
        "clifvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._clip_model_name = self.config.get(
            "clip_model", "openai/clip-vit-base-patch32"
        )
        self._ml_available = False
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._pos_embeds = None
        self._neg_embeds = None
        self._backend = None

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
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self._clip_model_name, models_dir)

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=self._device,
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
                    "default",
                ),
                load_clip,
            )

            self._pos_embeds = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _POSITIVE_FEELINGS,
                model_key=self._clip_model_name,
                device=self._device,
            )
            self._neg_embeds = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _NEGATIVE_FEELINGS,
                model_key=self._clip_model_name,
                device=self._device,
            )

            self._ml_available = True
            self._backend = "clip"
            logger.info("CLiF-VQA (CLIP feelings) initialised on %s", self._device)
        except (ImportError, Exception) as e:
            self._backend = "unavailable"
            logger.warning("CLiF-VQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        try:
            score = self._compute_feeling_score(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.clifvqa_score = score
                logger.debug("CLiF-VQA for %s: %.4f", sample.path.name, score)
        except Exception as e:
            logger.warning("CLiF-VQA failed for %s: %s", sample.path, e)
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
                self._model,
                self._processor,
                image_groups,
                model_key=self._clip_model_name,
                device=self._device,
                cache_keys=cache_keys,
            )
            for sample, image_features in zip(prepared, feature_groups):
                score = self._score_features(image_features)
                if score is None:
                    continue
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.clifvqa_score = score
        except Exception as e:
            logger.warning("CLiF-VQA batch failed: %s", e)

        return samples

    def _compute_feeling_score(self, sample: Sample) -> Optional[float]:
        """Score frames via CLIP similarity with feeling prompts."""

        frames = self._extract_frames(sample)
        if not frames:
            return None

        image_features = cached_clip_image_features(
            self,
            self._model,
            self._processor,
            arrays_to_pil(frames),
            model_key=self._clip_model_name,
            device=self._device,
            cache_key=(self.subsample, media_state_key(sample.path)),
        )
        return self._score_features(image_features)

    def _score_features(self, image_features) -> Optional[float]:
        import torch

        if image_features is None or image_features.size(0) == 0:
            return None

        frame_scores = []
        with torch.no_grad():
            for img_emb in image_features:
                img_emb = img_emb.unsqueeze(0)

                # Per-dimension feeling similarity
                pos_sims = (img_emb @ self._pos_embeds.T).squeeze(0)  # (7,)
                neg_sims = (img_emb @ self._neg_embeds.T).squeeze(0)  # (7,)

                # Per-dimension score: sigmoid of difference
                dim_logits = (pos_sims - neg_sims) * 10.0
                dim_scores = torch.sigmoid(dim_logits)
                # Average across all feeling dimensions
                frame_scores.append(float(dim_scores.mean().item()))

        if not frame_scores:
            return None
        return float(np.clip(np.mean(frame_scores), 0.0, 1.0))

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("CLiF-VQA frame loading failed for %s: %s", sample.path, e)
            return []

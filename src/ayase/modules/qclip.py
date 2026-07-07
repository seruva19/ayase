"""Q-CLIP --- VLM-Based VQA via Cross-Modal Quality Adaptation (2025).

This module implements a CLIP-IQA-style zero-shot quality signal: it scores
frames with a real, pretrained CLIP backbone (config-selectable via
``clip_model``) against quality-level text prompts and aggregates temporally.
It does NOT load the trained Q-CLIP cross-modal adaptation weights, so treat
``qclip_score`` as a CLIP-IQA-family proxy rather than the published model's
output.

qclip_score --- higher = better (0-1 range)
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
    media_state_key,
)

logger = logging.getLogger(__name__)

# Quality-level prompts for softmax scoring (modelled after CLIP-IQA)
_QUALITY_LEVELS = [
    ("excellent quality, perfectly sharp, superb detail", 1.0),
    ("good quality, mostly sharp, clear detail", 0.80),
    ("decent quality, reasonably clear", 0.60),
    ("fair quality, slightly blurry or noisy", 0.40),
    ("poor quality, blurry and noisy", 0.20),
    ("terrible quality, very blurry, heavy noise and artifacts", 0.0),
]


class QCLIPModule(PipelineModule):
    name = "qclip"
    description = "Q-CLIP VLM-based VQA (2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }
    metric_groups = {
        "qclip_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._clip_model_name = self.config.get(
            "clip_model", "openai/clip-vit-base-patch32"
        )
        self._ml_available = False
        self._backend = "unavailable"
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._quality_embeds = None
        self._quality_weights = None

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import torch
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

            # Pre-encode quality level prompts
            texts = [q[0] for q in _QUALITY_LEVELS]
            self._quality_weights = np.array([q[1] for q in _QUALITY_LEVELS])

            with torch.no_grad():
                text_inputs = self._processor(
                    text=texts, return_tensors="pt", padding=True
                ).to(self._device)
                self._quality_embeds = self._model.get_text_features(**text_inputs)
                self._quality_embeds = self._quality_embeds / self._quality_embeds.norm(
                    dim=-1, keepdim=True
                )

            self._ml_available = True
            self._backend = "clip"
            logger.info("Q-CLIP (CLIP quality) initialised on %s", self._device)
        except (ImportError, Exception) as e:
            self._backend = "unavailable"
            logger.warning("Q-CLIP setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            score = self._compute_quality_score(sample, frames)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.qclip_score = score
            logger.debug("Q-CLIP for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("Q-CLIP failed for %s: %s", sample.path, e)
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
                score = self._score_image_features(image_features)
                if score is None:
                    continue
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.qclip_score = score
        except Exception as e:
            logger.warning("Q-CLIP batch failed: %s", e)

        return samples

    def _compute_quality_score(self, sample: Sample, frames: List[np.ndarray]) -> Optional[float]:
        """Softmax over quality-level prompts, weighted average as final score."""

        pil_frames = arrays_to_pil(frames)
        image_features = cached_clip_image_features(
            self,
            self._model,
            self._processor,
            pil_frames,
            model_key=self._clip_model_name,
            device=self._device,
            cache_key=(self.subsample, media_state_key(sample.path)),
        )
        return self._score_image_features(image_features)

    def _score_image_features(self, image_features) -> Optional[float]:
        """Softmax over quality-level prompts, weighted average as final score."""
        if image_features is None or image_features.size(0) == 0:
            return None

        frame_scores = []
        for img_emb in image_features:
            # Cosine similarity with each quality level
            sims = (img_emb @ self._quality_embeds.T).cpu().numpy()
            # Temperature-scaled softmax
            exp_sims = np.exp((sims - sims.max()) * 100.0)
            probs = exp_sims / exp_sims.sum()
            # Weighted quality score
            score = float(np.dot(probs, self._quality_weights))
            frame_scores.append(score)

        if not frame_scores:
            return None

        # For video: also consider temporal consistency
        if len(frame_scores) > 1:
            # Penalise high variance in per-frame quality
            mean_score = float(np.mean(frame_scores))
            score_var = float(np.var(frame_scores))
            temporal_penalty = 1.0 / (1.0 + score_var * 10.0)
            return float(np.clip(mean_score * temporal_penalty, 0.0, 1.0))

        return float(np.clip(frame_scores[0], 0.0, 1.0))

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("Q-CLIP frame loading failed for %s: %s", sample.path, e)
            return []

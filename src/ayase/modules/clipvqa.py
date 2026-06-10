"""CLIPVQA — Video Quality Assessment via CLIP.

IEEE TIP 2024 — CLIP-based frame encoder with self-attention
for spatiotemporal quality. 37% better generalizability than
existing VQA methods across 8 datasets.

GitHub: https://github.com/GZHU-DVL/CLIPVQA

Backend tiers:
  1. **clipvqa** — native clipvqa package
  2. **CLIP** — CLIP features with self-attention temporal pooling

clipvqa_score — higher = better quality
"""

import logging
import numpy as np
from typing import List, Optional

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


class CLIPVQAModule(PipelineModule):
    name = "clipvqa"
    description = "CLIPVQA CLIP-based spatiotemporal VQA (TIP 2024)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }
    metric_groups = {
        "clipvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
        self._model = None
        self._clip_model = None
        self._clip_processor = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None
        self._quality_text_embeds = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import clipvqa
            self._model = clipvqa
            self._backend = "native"
            self._ml_available = True
            logger.info("CLIPVQA (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: CLIP
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
            self._quality_text_embeds = cached_clip_text_features(
                self,
                self._clip_model,
                self._clip_processor,
                [
                    "a perfectly clear high quality image",
                    "a very blurry low quality image",
                ],
                model_key=self._clip_model_name,
                device=self._device,
            )
            self._backend = "clip"
            self._ml_available = True
            logger.info(f"CLIPVQA (CLIP) initialised on {self._device}")
            return
        except (ImportError, Exception):
            pass

        logger.warning("CLIPVQA unavailable: install clipvqa or transformers")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            if self._backend == "native":
                score = float(self._model.predict(str(sample.path)))
            elif self._backend == "clip":
                score = self._process_clip(sample)
            else:
                return sample

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.clipvqa_score = score

        except Exception as e:
            logger.warning(f"CLIPVQA failed for {sample.path}: {e}")

        return sample

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if not self._ml_available:
            return samples
        if self._backend != "clip":
            return super().process_batch(samples)

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
                sample.quality_metrics.clipvqa_score = score
        except Exception as e:
            logger.warning("CLIPVQA batch failed: %s", e)

        return samples

    def _process_clip(self, sample: Sample) -> Optional[float]:
        """CLIP features + self-attention temporal pooling approximation."""

        frames = self._extract_frames(sample)
        if not frames:
            return None

        image_features = cached_clip_image_features(
            self,
            self._clip_model,
            self._clip_processor,
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

        logits = image_features @ self._quality_text_embeds.T
        if hasattr(self._clip_model, "logit_scale"):
            logits = logits * self._clip_model.logit_scale.exp()
        quality_scores = torch.softmax(logits, dim=-1)[:, 0].cpu().numpy()
        embeddings = image_features.detach().cpu().numpy()

        # Self-attention-like temporal weighting
        if len(embeddings) > 1:
            # Compute attention weights via dot product similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / (norms + 1e-8)
            attention = normalized @ normalized.T
            weights = np.mean(attention, axis=1)
            weights = np.exp(weights) / np.sum(np.exp(weights))
            score = float(np.dot(weights, quality_scores))
        else:
            score = float(quality_scores[0]) if len(quality_scores) else None

        return score

    def _extract_frames(self, sample: Sample):
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("CLIPVQA frame loading failed for %s: %s", sample.path, e)
            return []

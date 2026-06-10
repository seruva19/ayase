"""ChronoMagic-Bench module — NeurIPS 2024 (arXiv:2406.18522).

Two metrics for time-lapse video quality:
  - **MTScore** (Metamorphic Temporal): smoothness of temporal progression
  - **CHScore** (Chrono-Hallucination): detection of temporal hallucinations

Backend tier:
  1. **CLIP** — CLIP ViT-B/32 temporal embedding analysis

Video-only: returns None for images.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.runtime import cached_clip_image_features, media_state_key

logger = logging.getLogger(__name__)


class ChronoMagicModule(PipelineModule):
    name = "chronomagic"
    description = "ChronoMagic-Bench MTScore + CHScore (CLIP)"
    default_config = {
        "subsample": 16,
        "clip_model": "openai/clip-vit-base-patch32",
        "hallucination_threshold": 2.0,
    }
    metric_groups = {
        "chronomagic_ch_score": "temporal",
        "chronomagic_mt_score": "temporal",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 16)
        self.clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
        self._ml_available = False
        self._clip_model = None
        self._clip_processor = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: CLIP
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
            resolved = resolve_model_path(self.clip_model_name, models_dir)

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
            self._ml_available = True
            logger.info("ChronoMagic loaded CLIP on %s", self._device)
            return
        except Exception as e:
            logger.info("CLIP unavailable for ChronoMagic: %s", e)

        logger.warning("ChronoMagic unavailable: CLIP not installed")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        if not sample.is_video:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            mt_score, ch_score = self._compute_clip(sample)

            if mt_score is not None:
                sample.quality_metrics.chronomagic_mt_score = mt_score
            if ch_score is not None:
                sample.quality_metrics.chronomagic_ch_score = ch_score

        except Exception as e:
            logger.warning("ChronoMagic processing failed: %s", e)

        return sample

    # ------------------------------------------------------------------ #
    # Shared frame extraction                                              #
    # ------------------------------------------------------------------ #

    def _extract_frames(self, sample: Sample) -> list:
        frames = sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        if len(frames) < 4:
            return []
        return frames

    # ------------------------------------------------------------------ #
    # CLIP temporal analysis                                               #
    # ------------------------------------------------------------------ #

    def _compute_clip(self, sample: Sample) -> Tuple[Optional[float], Optional[float]]:
        frames = self._extract_frames(sample)
        if len(frames) < 4:
            return None, None

        embeddings = cached_clip_image_features(
            self,
            self._clip_model,
            self._clip_processor,
            arrays_to_pil(frames),
            model_key=self.clip_model_name,
            device=self._device,
            cache_key=(self.subsample, media_state_key(sample.path)),
        ).detach().float().cpu().numpy()

        # MTScore: temporal gradient smoothness
        gradients = np.diff(embeddings, axis=0)  # [T-1, D]
        gradient_norms = np.linalg.norm(gradients, axis=1)  # [T-1]

        # Smoothness = low variance of gradient norms (consistent rate of change)
        if gradient_norms.std() > 1e-8:
            gradient_cv = gradient_norms.std() / (gradient_norms.mean() + 1e-8)
            gradient_smoothness = 1.0 / (1.0 + gradient_cv)
        else:
            gradient_smoothness = 1.0

        # Progression quality: monotonic progression in embedding space
        cosine_sims = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            cosine_sims.append(sim)
        cosine_sims = np.array(cosine_sims)

        # Good progression = consistently high but not perfect similarity
        progression_quality = float(np.mean(cosine_sims))
        progression_quality = min(max(progression_quality, 0.0), 1.0)

        mt_score = 0.5 * gradient_smoothness + 0.5 * progression_quality

        # CHScore: detect abrupt embedding spikes (hallucination frames)
        threshold = self.config.get("hallucination_threshold", 2.0)
        mean_norm = gradient_norms.mean()
        std_norm = gradient_norms.std() + 1e-8
        hallucination_mask = gradient_norms > (mean_norm + threshold * std_norm)
        hallucination_frac = float(np.mean(hallucination_mask))
        ch_score = 1.0 - hallucination_frac  # Higher = fewer hallucinations

        return (
            float(np.clip(mt_score, 0.0, 1.0)),
            float(np.clip(ch_score, 0.0, 1.0)),
        )

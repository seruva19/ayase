"""CRAVE --- Content-Rich AIGC Video Evaluator (2025).

GitHub: https://github.com/littlespray/CRAVE

Designed for Sora-era videos. Uses CLIP for content understanding +
temporal consistency.  Scores based on:
  - Content richness: diversity of CLIP features across frames
  - Visual quality: CLIP similarity with quality prompts
  - Temporal coherence: smoothness of CLIP embeddings over time

crave_score --- higher = better quality (0-1 range)
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

_QUALITY_POS = [
    "a high quality video with rich content",
    "a detailed video with diverse visual elements",
    "a sharp clear professional video",
]
_QUALITY_NEG = [
    "a low quality video with poor content",
    "a repetitive monotonous video",
    "a blurry noisy amateur video",
]


class CRAVEModule(PipelineModule):
    name = "crave"
    description = "CRAVE content-rich AIGC video evaluator (2025)"
    default_config = {
        "subsample": 12,
        "clip_model": "openai/clip-vit-base-patch32",
        "quality_weight": 0.35,
        "richness_weight": 0.35,
        "coherence_weight": 0.30,
    }
    metric_groups = {
        "crave_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 12)
        self._clip_model_name = self.config.get(
            "clip_model", "openai/clip-vit-base-patch32"
        )
        self._w_quality = self.config.get("quality_weight", 0.35)
        self._w_richness = self.config.get("richness_weight", 0.35)
        self._w_coherence = self.config.get("coherence_weight", 0.30)
        self._ml_available = False
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._quality_pos = None
        self._quality_neg = None

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

            self._quality_pos = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _QUALITY_POS,
                model_key=self._clip_model_name,
                device=self._device,
            )
            self._quality_neg = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _QUALITY_NEG,
                model_key=self._clip_model_name,
                device=self._device,
            )

            self._ml_available = True
            logger.info("CRAVE (CLIP content-rich) initialised on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("CRAVE setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        try:
            frames = self._extract_frames(sample)
            if len(frames) < 2:
                return sample

            score = self._compute_score(sample, frames)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.crave_score = score
            logger.debug("CRAVE for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("CRAVE failed for %s: %s", sample.path, e)
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
                frames = self._extract_frames(sample)
                if len(frames) < 2:
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
            for sample, frame_stack in zip(prepared, feature_groups):
                score = self._score_features(frame_stack)
                if score is None:
                    continue
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.crave_score = score
        except Exception as e:
            logger.warning("CRAVE batch failed: %s", e)

        return samples

    def _compute_score(self, sample: Sample, frames: List[np.ndarray]) -> Optional[float]:
        frame_stack = cached_clip_image_features(
            self,
            self._model,
            self._processor,
            arrays_to_pil(frames),
            model_key=self._clip_model_name,
            device=self._device,
            cache_key=(self.subsample, media_state_key(sample.path)),
        )
        return self._score_features(frame_stack)

    def _score_features(self, frame_stack) -> Optional[float]:
        import torch

        if frame_stack is None or frame_stack.size(0) < 2:
            return None

        # --- Visual Quality: CLIP similarity with quality prompts ---
        with torch.no_grad():
            pos_sim = (frame_stack @ self._quality_pos.T).mean(dim=-1)
            neg_sim = (frame_stack @ self._quality_neg.T).mean(dim=-1)
            q_logits = (pos_sim - neg_sim) * 10.0
            quality = float(torch.sigmoid(q_logits).mean().item())

        # --- Content Richness: diversity of CLIP features across frames ---
        # Higher diversity of embeddings = richer content
        with torch.no_grad():
            # Pairwise cosine similarity matrix
            sim_matrix = frame_stack @ frame_stack.T  # (N, N)
            n = sim_matrix.shape[0]
            # Mask diagonal
            mask = ~torch.eye(n, dtype=torch.bool, device=self._device)
            mean_pairwise_sim = sim_matrix[mask].mean().item()
            # Lower mean similarity = more diverse content
            # Map: sim=1.0 (no diversity) -> 0, sim=0.5 -> 1
            richness = float(np.clip(1.0 - (mean_pairwise_sim - 0.5) * 2.0, 0.0, 1.0))

        # --- Temporal Coherence: smoothness of consecutive embeddings ---
        cosine_sims = []
        for i in range(frame_stack.size(0) - 1):
            sim = torch.nn.functional.cosine_similarity(
                frame_stack[i].unsqueeze(0), frame_stack[i + 1].unsqueeze(0)
            ).item()
            cosine_sims.append(sim)
        mean_sim = float(np.mean(cosine_sims))
        var_sim = float(np.var(cosine_sims))
        # High mean + low variance = smooth transitions
        coherence = float(np.clip(
            mean_sim * (1.0 / (1.0 + var_sim * 100.0)), 0.0, 1.0
        ))

        score = (
            self._w_quality * quality
            + self._w_richness * richness
            + self._w_coherence * coherence
        )
        return float(np.clip(score, 0.0, 1.0))

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("CRAVE frame loading failed for %s: %s", sample.path, e)
            return []

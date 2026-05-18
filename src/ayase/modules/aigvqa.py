"""AIGVQA --- Multi-Dimensional AI-Generated VQA (ICCVW 2025).

GitHub: https://github.com/IntMeGroup/AIGVQA

Multi-dimensional quality scoring using CLIP features:
  - Spatial quality via CLIP quality-prompt similarity
  - Temporal consistency via CLIP frame embedding coherence
  - Aesthetic quality via CLIP aesthetic-prompt similarity

Final score is a weighted combination of all three dimensions.

aigvqa_score --- higher = better (0-1 range)
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

_SPATIAL_POS = [
    "a sharp image with fine details",
    "a clear frame with good resolution",
]
_SPATIAL_NEG = [
    "a blurry image lacking detail",
    "a pixelated low resolution frame",
]
_AESTHETIC_POS = [
    "a beautiful and visually appealing scene",
    "an image with good color and composition",
]
_AESTHETIC_NEG = [
    "an ugly and unpleasant scene",
    "an image with bad colors and poor composition",
]


class AIGVQAModule(PipelineModule):
    name = "aigvqa"
    description = "AIGVQA multi-dimensional AIGC VQA (ICCVW 2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
        "spatial_weight": 0.4,
        "temporal_weight": 0.3,
        "aesthetic_weight": 0.3,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._clip_model_name = self.config.get(
            "clip_model", "openai/clip-vit-base-patch32"
        )
        self._w_spatial = self.config.get("spatial_weight", 0.4)
        self._w_temporal = self.config.get("temporal_weight", 0.3)
        self._w_aesthetic = self.config.get("aesthetic_weight", 0.3)
        self._ml_available = False
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._spatial_pos = None
        self._spatial_neg = None
        self._aes_pos = None
        self._aes_neg = None

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

            self._spatial_pos = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _SPATIAL_POS,
                model_key=self._clip_model_name,
                device=self._device,
            )
            self._spatial_neg = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _SPATIAL_NEG,
                model_key=self._clip_model_name,
                device=self._device,
            )
            self._aes_pos = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _AESTHETIC_POS,
                model_key=self._clip_model_name,
                device=self._device,
            )
            self._aes_neg = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _AESTHETIC_NEG,
                model_key=self._clip_model_name,
                device=self._device,
            )

            self._ml_available = True
            logger.info("AIGVQA (CLIP multi-dim) initialised on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("AIGVQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            score = self._compute_score(sample, frames)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.aigvqa_score = score
            logger.debug("AIGVQA for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("AIGVQA failed for %s: %s", sample.path, e)
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
            for sample, frame_stack in zip(prepared, feature_groups):
                score = self._score_features(frame_stack)
                if score is None:
                    continue
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.aigvqa_score = score
        except Exception as e:
            logger.warning("AIGVQA batch failed: %s", e)

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

        if frame_stack is None or frame_stack.size(0) == 0:
            return None

        # --- Spatial quality ---
        with torch.no_grad():
            sp_pos = (frame_stack @ self._spatial_pos.T).mean(dim=-1)
            sp_neg = (frame_stack @ self._spatial_neg.T).mean(dim=-1)
            sp_logits = (sp_pos - sp_neg) * 10.0
            spatial = float(torch.sigmoid(sp_logits).mean().item())

        # --- Aesthetic quality ---
        with torch.no_grad():
            ae_pos = (frame_stack @ self._aes_pos.T).mean(dim=-1)
            ae_neg = (frame_stack @ self._aes_neg.T).mean(dim=-1)
            ae_logits = (ae_pos - ae_neg) * 10.0
            aesthetic = float(torch.sigmoid(ae_logits).mean().item())

        # --- Temporal consistency: cosine similarity between consecutive frames ---
        if frame_stack.size(0) > 1:
            cosine_sims = []
            for i in range(frame_stack.size(0) - 1):
                sim = torch.nn.functional.cosine_similarity(
                    frame_stack[i].unsqueeze(0), frame_stack[i + 1].unsqueeze(0)
                ).item()
                cosine_sims.append(sim)
            # Combine mean consistency and smoothness (low variance)
            mean_sim = float(np.mean(cosine_sims))
            var_sim = float(np.var(cosine_sims))
            temporal = float(np.clip(
                mean_sim * (1.0 / (1.0 + var_sim * 50.0)), 0.0, 1.0
            ))
        else:
            temporal = 1.0  # Single frame: perfect temporal consistency

        score = (
            self._w_spatial * spatial
            + self._w_temporal * temporal
            + self._w_aesthetic * aesthetic
        )
        return float(np.clip(score, 0.0, 1.0))

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("AIGVQA frame loading failed for %s: %s", sample.path, e)
            return []

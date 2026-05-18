"""VideoReward --- Kling Multi-Dimensional Reward Model (NeurIPS 2025).

HuggingFace: https://huggingface.co/KlingTeam/VideoReward

Uses CLIP backbone to encode video frames and compute multi-dimensional
reward scores:
  - Visual Quality (VQ): CLIP similarity with quality text prompts
  - Motion Quality (MQ): temporal consistency of CLIP frame embeddings
  - Text Alignment (TA): CLIP cosine similarity with the sample caption

videoreward_vq, videoreward_mq, videoreward_ta --- all higher = better (0-1)
"""

import logging
from typing import Dict, List, Optional

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

_QUALITY_PROMPTS = [
    "a high quality video with sharp details",
    "a clear video with good exposure",
    "a professional video with vivid colors",
]
_LOW_QUALITY_PROMPTS = [
    "a blurry low quality video",
    "a noisy distorted video",
]


class VideoRewardModule(PipelineModule):
    name = "videoreward"
    description = "VideoReward Kling multi-dim reward model (NeurIPS 2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
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
        self._quality_embeds = None
        self._low_quality_embeds = None

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

            # Pre-encode quality prompts
            self._quality_embeds = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _QUALITY_PROMPTS,
                model_key=self._clip_model_name,
                device=self._device,
            )
            self._low_quality_embeds = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _LOW_QUALITY_PROMPTS,
                model_key=self._clip_model_name,
                device=self._device,
            )

            self._ml_available = True
            logger.info("VideoReward (CLIP) initialised on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("VideoReward setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            scores = self._compute_rewards(frames, sample)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.videoreward_vq = scores["vq"]
            sample.quality_metrics.videoreward_mq = scores["mq"]
            sample.quality_metrics.videoreward_ta = scores["ta"]

            logger.debug(
                "VideoReward for %s: vq=%.3f mq=%.3f ta=%.3f",
                sample.path.name,
                scores["vq"],
                scores["mq"],
                scores["ta"],
            )
        except Exception as e:
            logger.warning("VideoReward failed for %s: %s", sample.path, e)
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
                if frame_stack is None or frame_stack.size(0) == 0:
                    continue
                scores = self._compute_rewards_from_features(frame_stack, sample)
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.videoreward_vq = scores["vq"]
                sample.quality_metrics.videoreward_mq = scores["mq"]
                sample.quality_metrics.videoreward_ta = scores["ta"]
        except Exception as e:
            logger.warning("VideoReward batch failed: %s", e)

        return samples

    def _compute_rewards(
        self, frames: List[np.ndarray], sample: Sample
    ) -> Dict[str, float]:
        # Encode all frames
        frame_stack = cached_clip_image_features(
            self,
            self._model,
            self._processor,
            arrays_to_pil(frames),
            model_key=self._clip_model_name,
            device=self._device,
            cache_key=(self.subsample, media_state_key(sample.path)),
        )
        return self._compute_rewards_from_features(frame_stack, sample)

    def _compute_rewards_from_features(
        self,
        frame_stack,
        sample: Sample,
    ) -> Dict[str, float]:
        import torch

        # --- Visual Quality: CLIP similarity with quality prompts ---
        with torch.no_grad():
            pos_sims = (frame_stack @ self._quality_embeds.T).mean(dim=-1)
            neg_sims = (frame_stack @ self._low_quality_embeds.T).mean(dim=-1)
            vq_logits = (pos_sims - neg_sims) * 10.0
            vq_scores = torch.sigmoid(vq_logits)
            vq = float(vq_scores.mean().item())

        # --- Motion Quality: temporal smoothness of frame embeddings ---
        if frame_stack.size(0) > 1:
            cosine_sims = []
            for i in range(frame_stack.size(0) - 1):
                sim = torch.nn.functional.cosine_similarity(
                    frame_stack[i].unsqueeze(0), frame_stack[i + 1].unsqueeze(0)
                ).item()
                cosine_sims.append(sim)
            # High mean similarity + low variance = smooth motion
            mean_sim = float(np.mean(cosine_sims))
            var_sim = float(np.var(cosine_sims))
            mq = float(np.clip(mean_sim * (1.0 / (1.0 + var_sim * 100.0)), 0.0, 1.0))
        else:
            mq = 0.5

        # --- Text Alignment: CLIP cosine with caption ---
        caption = getattr(sample, "caption", None)
        caption_text = None
        if caption and hasattr(caption, "text"):
            caption_text = caption.text
        elif isinstance(caption, str):
            caption_text = caption

        if caption_text:
            with torch.no_grad():
                text_emb = cached_clip_text_features(
                    self,
                    self._model,
                    self._processor,
                    [caption_text],
                    model_key=self._clip_model_name,
                    device=self._device,
                )
                sims = (frame_stack @ text_emb.T).squeeze(-1)
                # Map from CLIP cosine range (~0.15-0.35) to 0-1
                ta = float(torch.clamp((sims.mean() - 0.15) / 0.20, 0.0, 1.0).item())
        else:
            # No caption: use mid-range score
            ta = 0.5

        return {
            "vq": float(np.clip(vq, 0.0, 1.0)),
            "mq": float(np.clip(mq, 0.0, 1.0)),
            "ta": float(np.clip(ta, 0.0, 1.0)),
        }

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("VideoReward frame loading failed for %s: %s", sample.path, e)
            return []

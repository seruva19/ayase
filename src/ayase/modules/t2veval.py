"""T2VEval --- Text-to-Video Generated Video Evaluation (2025).

Evaluates text-video consistency and realness using CLIP:
  - Alignment: CLIP cosine similarity between frames and caption
  - Realness: CLIP discriminative features measuring distance from
    "real" vs "generated" text embeddings
  - Quality: CLIP quality-prompt scoring

t2veval_score --- higher = better (0-1 range)
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

_REAL_PROMPTS = [
    "a real natural video",
    "an authentic video captured by a camera",
    "a realistic video of a real scene",
]
_FAKE_PROMPTS = [
    "an AI generated synthetic video",
    "a computer generated artificial video",
    "a fake video produced by a machine",
]
_QUALITY_POS = [
    "a high quality well-produced video",
    "a sharp clear video with good detail",
]
_QUALITY_NEG = [
    "a low quality poorly produced video",
    "a blurry noisy video with artifacts",
]


class T2VEvalModule(PipelineModule):
    name = "t2veval"
    description = "T2VEval text-video consistency+realness (2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
        "alignment_weight": 0.35,
        "realness_weight": 0.35,
        "quality_weight": 0.30,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._clip_model_name = self.config.get(
            "clip_model", "openai/clip-vit-base-patch32"
        )
        self._w_align = self.config.get("alignment_weight", 0.35)
        self._w_real = self.config.get("realness_weight", 0.35)
        self._w_quality = self.config.get("quality_weight", 0.30)
        self._ml_available = False
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._real_embeds = None
        self._fake_embeds = None
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

            self._real_embeds = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _REAL_PROMPTS,
                model_key=self._clip_model_name,
                device=self._device,
            )
            self._fake_embeds = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _FAKE_PROMPTS,
                model_key=self._clip_model_name,
                device=self._device,
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
            logger.info("T2VEval (CLIP) initialised on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("T2VEval setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            score = self._compute_score(frames, sample)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.t2veval_score = score
            logger.debug("T2VEval for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("T2VEval failed for %s: %s", sample.path, e)
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
                score = self._score_features(frame_stack, sample)
                if score is None:
                    continue
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.t2veval_score = score
        except Exception as e:
            logger.warning("T2VEval batch failed: %s", e)

        return samples

    def _compute_score(
        self, frames: List[np.ndarray], sample: Sample
    ) -> Optional[float]:
        frame_stack = cached_clip_image_features(
            self,
            self._model,
            self._processor,
            arrays_to_pil(frames),
            model_key=self._clip_model_name,
            device=self._device,
            cache_key=(self.subsample, media_state_key(sample.path)),
        )
        return self._score_features(frame_stack, sample)

    def _score_features(self, frame_stack, sample: Sample) -> Optional[float]:
        import torch

        if frame_stack is None or frame_stack.size(0) == 0:
            return None

        # --- Realness: distance from "real" vs "generated" prompts ---
        with torch.no_grad():
            real_sim = (frame_stack @ self._real_embeds.T).mean(dim=-1)
            fake_sim = (frame_stack @ self._fake_embeds.T).mean(dim=-1)
            real_logits = (real_sim - fake_sim) * 10.0
            realness = float(torch.sigmoid(real_logits).mean().item())

        # --- Visual Quality ---
        with torch.no_grad():
            qp_sim = (frame_stack @ self._quality_pos.T).mean(dim=-1)
            qn_sim = (frame_stack @ self._quality_neg.T).mean(dim=-1)
            q_logits = (qp_sim - qn_sim) * 10.0
            quality = float(torch.sigmoid(q_logits).mean().item())

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
                    truncation=True,
                )
                sims = (frame_stack @ text_emb.T).squeeze(-1)
                # Map from typical CLIP cosine range to 0-1
                alignment = float(
                    torch.clamp((sims.mean() - 0.15) / 0.20, 0.0, 1.0).item()
                )
        else:
            # Without caption: rely on quality + realness only
            alignment = 0.5

        score = (
            self._w_align * alignment
            + self._w_real * realness
            + self._w_quality * quality
        )
        return float(np.clip(score, 0.0, 1.0))

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("T2VEval frame loading failed for %s: %s", sample.path, e)
            return []

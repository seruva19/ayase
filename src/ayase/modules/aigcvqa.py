"""AIGC-VQA --- Holistic Perception for AIGC Video Quality (CVPRW 2024).

3-branch assessment using CLIP features:
  - Technical quality: CLIP similarity with technical quality prompts
  - Aesthetic quality: CLIP similarity with aesthetic quality prompts
  - Text-video alignment: CLIP cosine similarity with sample caption

aigcvqa_technical, aigcvqa_aesthetic, aigcvqa_alignment --- all 0-1, higher = better
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

_TECHNICAL_POS = [
    "a sharp video with clear details",
    "a video with no noise or compression artifacts",
    "a well-exposed video with good dynamic range",
]
_TECHNICAL_NEG = [
    "a blurry video with noise",
    "a video with compression artifacts and distortion",
    "an overexposed or underexposed video",
]
_AESTHETIC_POS = [
    "a beautiful video with good composition",
    "a visually pleasing video with harmonious colors",
    "an artistically composed video",
]
_AESTHETIC_NEG = [
    "an ugly video with bad composition",
    "a visually unpleasant video with clashing colors",
    "a poorly composed video",
]


class AIGCVQAModule(PipelineModule):
    name = "aigcvqa"
    description = "AIGC-VQA holistic 3-branch AIGC perception (CVPRW 2024)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }
    metric_groups = {
        "aigcvqa_aesthetic": "nr_quality",
        "aigcvqa_alignment": "alignment",
        "aigcvqa_technical": "nr_quality",
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
        # Pre-encoded prompt embeddings
        self._tech_pos = None
        self._tech_neg = None
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

            self._tech_pos = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _TECHNICAL_POS,
                model_key=self._clip_model_name,
                device=self._device,
            )
            self._tech_neg = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _TECHNICAL_NEG,
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
            logger.info("AIGC-VQA (CLIP 3-branch) initialised on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("AIGC-VQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            scores = self._compute_scores(frames, sample)
            if scores is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.aigcvqa_technical = scores["technical"]
            sample.quality_metrics.aigcvqa_aesthetic = scores["aesthetic"]
            if scores["alignment"] is not None:
                sample.quality_metrics.aigcvqa_alignment = scores["alignment"]

            logger.debug(
                "AIGC-VQA for %s: tech=%.3f aes=%.3f align=%s",
                sample.path.name,
                scores["technical"],
                scores["aesthetic"],
                scores["alignment"],
            )
        except Exception as e:
            logger.warning("AIGC-VQA failed for %s: %s", sample.path, e)
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
                scores = self._score_features(frame_stack, sample)
                if scores is None:
                    continue
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.aigcvqa_technical = scores["technical"]
                sample.quality_metrics.aigcvqa_aesthetic = scores["aesthetic"]
                if scores["alignment"] is not None:
                    sample.quality_metrics.aigcvqa_alignment = scores["alignment"]
        except Exception as e:
            logger.warning("AIGC-VQA batch failed: %s", e)

        return samples

    def _compute_scores(
        self, frames: List[np.ndarray], sample: Sample
    ) -> Optional[Dict[str, Optional[float]]]:
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

    def _score_features(
        self, frame_stack, sample: Sample
    ) -> Optional[Dict[str, Optional[float]]]:
        import torch
        if frame_stack is None or frame_stack.size(0) == 0:
            return None

        # --- Technical branch ---
        with torch.no_grad():
            tech_pos_sim = (frame_stack @ self._tech_pos.T).mean(dim=-1)
            tech_neg_sim = (frame_stack @ self._tech_neg.T).mean(dim=-1)
            tech_logits = (tech_pos_sim - tech_neg_sim) * 10.0
            technical = float(torch.sigmoid(tech_logits).mean().item())

        # --- Aesthetic branch ---
        with torch.no_grad():
            aes_pos_sim = (frame_stack @ self._aes_pos.T).mean(dim=-1)
            aes_neg_sim = (frame_stack @ self._aes_neg.T).mean(dim=-1)
            aes_logits = (aes_pos_sim - aes_neg_sim) * 10.0
            aesthetic = float(torch.sigmoid(aes_logits).mean().item())

        # --- Alignment branch: CLIP cosine with caption ---
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
                # Map from CLIP cosine range (~0.15-0.35) to 0-1
                alignment = float(
                    torch.clamp((sims.mean() - 0.15) / 0.20, 0.0, 1.0).item()
                )
        else:
            alignment = None  # No caption available: leave alignment unset

        return {
            "technical": float(np.clip(technical, 0.0, 1.0)),
            "aesthetic": float(np.clip(aesthetic, 0.0, 1.0)),
            "alignment": alignment,
        }

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("AIGC-VQA frame loading failed for %s: %s", sample.path, e)
            return []

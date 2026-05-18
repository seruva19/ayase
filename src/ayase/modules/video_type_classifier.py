"""Video type classifier module.

From NVIDIA Curator's video type classification concept.
Classifies content as real-world, animated, game, abstract, etc.
Uses CLIP zero-shot classification.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_clip_image_feature_groups,
    cached_clip_image_features,
    cached_clip_text_features,
    media_state_key,
)

logger = logging.getLogger(__name__)

VIDEO_TYPE_LABELS = [
    "real-world photography",
    "animated cartoon or anime",
    "video game footage",
    "abstract visual pattern",
    "screen recording or presentation",
    "text-heavy or document",
]

VIDEO_TYPE_SHORT = [
    "real",
    "animated",
    "game",
    "abstract",
    "screen_recording",
    "text_heavy",
]


class VideoTypeClassifierModule(PipelineModule):
    name = "video_type_classifier"
    description = "CLIP zero-shot video content type classification"
    default_config = {
        "subsample": 4,
        "clip_model": "openai/clip-vit-base-patch32",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._text_features = None

    def setup(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(model_name, models_dir)

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
            self._text_features = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                VIDEO_TYPE_LABELS,
                model_key=model_name,
                device=self._device,
            )
            self._ml_available = True
            logger.info("CLIP model loaded for video type classification on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("Video type classifier unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        try:
            image_features = self._encode_sample(sample)
            if image_features is None or image_features.size(0) == 0:
                return sample

            self._apply_type(sample, image_features)
        except Exception as e:
            logger.warning("Video type classification failed: %s", e)
        return sample

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if not self._ml_available:
            for sample in samples:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
            return samples

        try:
            prepared = []
            image_groups = []
            cache_keys = []
            subsample = self.config.get("subsample", 4)
            for sample in samples:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                frames = self._extract_frames(sample)
                if not frames:
                    continue
                prepared.append(sample)
                image_groups.append(arrays_to_pil(frames))
                cache_keys.append((subsample, media_state_key(sample.path)))

            if not prepared:
                return samples

            model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
            feature_groups = cached_clip_image_feature_groups(
                self,
                self._model,
                self._processor,
                image_groups,
                model_key=model_name,
                device=self._device,
                cache_keys=cache_keys,
            )
            for sample, image_features in zip(prepared, feature_groups):
                self._apply_type(sample, image_features)
        except Exception as e:
            logger.warning("Video type classification batch failed: %s", e)

        return samples

    def _extract_frames(self, sample: Sample):
        try:
            subsample = self.config.get("subsample", 4)
            return sample_frames(sample.path, max_frames=subsample, color="rgb")
        except Exception as e:
            logger.debug("Video type frame loading failed for %s: %s", sample.path, e)
            return []

    def _encode_sample(self, sample: Sample):
        frames = self._extract_frames(sample)
        if not frames:
            return None
        model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
        subsample = self.config.get("subsample", 4)
        return cached_clip_image_features(
            self,
            self._model,
            self._processor,
            arrays_to_pil(frames),
            model_key=model_name,
            device=self._device,
            cache_key=(subsample, media_state_key(sample.path)),
        )

    def _apply_type(self, sample: Sample, image_features) -> None:
        import torch

        if image_features is None or image_features.size(0) == 0:
            return
        logits = image_features @ self._text_features.T
        if hasattr(self._model, "logit_scale"):
            logits = logits * self._model.logit_scale.exp()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        votes = probs.mean(axis=0)
        best_idx = int(np.argmax(votes))
        sample.quality_metrics.video_type = VIDEO_TYPE_SHORT[best_idx]
        sample.quality_metrics.video_type_confidence = float(votes[best_idx])

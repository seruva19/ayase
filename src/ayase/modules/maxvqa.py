"""MaxVQA — Explainable VQA via Language-Prompted CLIP.

ACM MM 2023 Oral — language-prompted VQA using modified CLIP
for explainable quality scoring with MaxWell dataset.

GitHub: https://github.com/VQAssessment/ExplainableVQA

maxvqa_score — higher = better quality
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


class MaxVQAModule(PipelineModule):
    name = "maxvqa"
    description = "MaxVQA explainable language-prompted VQA (ACM MM 2023)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
        self._model = None
        self._clip_model = None
        self._clip_processor = None
        self._quality_text_features = None
        self._backend = None
        self._ml_available = False
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: native maxvqa package
        try:
            import maxvqa
            self._model = maxvqa
            self._backend = "native"
            self._ml_available = True
            logger.info("MaxVQA (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: CLIP-based quality scoring
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
            self._quality_text_features = cached_clip_text_features(
                self,
                self._clip_model,
                self._clip_processor,
                self._quality_texts(),
                model_key=self._clip_model_name,
                device=self._device,
                cache_key=("maxvqa_quality_texts",),
            )
            self._backend = "clip"
            self._ml_available = True
            logger.info(f"MaxVQA (CLIP) initialised on {self._device}")
            return
        except (ImportError, Exception):
            pass

        logger.warning("MaxVQA: no backend available (install maxvqa or transformers)")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            if self._backend == "native":
                score = self._process_native(sample)
            elif self._backend == "clip":
                score = self._process_clip(sample)
            else:
                return sample

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.maxvqa_score = score

        except Exception as e:
            logger.warning(f"MaxVQA failed for {sample.path}: {e}")

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
                score = self._score_clip_features(image_features)
                if score is None:
                    continue
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.maxvqa_score = score
        except Exception as e:
            logger.warning("MaxVQA batch failed: %s", e)
        return samples

    def _process_native(self, sample: Sample) -> Optional[float]:
        return float(self._model.predict(str(sample.path)))

    def _process_clip(self, sample: Sample) -> Optional[float]:
        """CLIP-based: cosine similarity with quality anchor texts."""
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
        return self._score_clip_features(image_features)

    @staticmethod
    def _quality_texts() -> List[str]:
        return [
            "a high quality, sharp, well-lit video frame",
            "a low quality, blurry, poorly-lit video frame",
        ]

    def _score_clip_features(self, image_features) -> Optional[float]:
        if image_features is None or image_features.size(0) == 0:
            return None
        scale = getattr(self._clip_model, "logit_scale", None)
        if scale is not None:
            logits = (image_features @ self._quality_text_features.T) * scale.exp()
        else:
            logits = image_features @ self._quality_text_features.T
        probs = logits.softmax(dim=-1)
        scores = probs[:, 0].detach().float().cpu().numpy()

        return float(np.mean(scores))

    def _extract_frames(self, sample: Sample):
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("MaxVQA frame loading failed for %s: %s", sample.path, e)
            return []

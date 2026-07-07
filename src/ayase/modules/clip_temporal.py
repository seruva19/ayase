"""CLIP Temporal Consistency + Face Consistency — EvalCrafter metrics #5 and #7.

- clip_temp_score: Cosine similarity between CLIP embeddings of consecutive
  frames, averaged across all pairs.  Measures temporal smoothness.
- face_consistency_score: Cosine similarity between consecutive frame pairs in
  CLIP space, averaged across the video.  Measures identity/appearance stability
  (EvalCrafter face_consistency).
"""

import logging

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_clip_image_feature_groups,
    cached_clip_image_features,
    media_state_key,
)

logger = logging.getLogger(__name__)


class CLIPTemporalModule(PipelineModule):
    name = "clip_temporal"
    description = "CLIP temporal consistency + face/identity consistency (EvalCrafter clip_temp & face_consistency)"
    default_config = {
        "model_name": "openai/clip-vit-base-patch32",
        "max_frames": 32,
        "temp_threshold": 0.90,
        "face_threshold": 0.85,
    }
    metric_groups = {
        "clip_temp": "temporal",
        "face_consistency": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "openai/clip-vit-base-patch32")
        self.max_frames = self.config.get("max_frames", 32)
        self.temp_threshold = self.config.get("temp_threshold", 0.90)
        self.face_threshold = self.config.get("face_threshold", 0.85)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            logger.info(f"Loading CLIP for temporal consistency on {self._device}...")
            from ayase.config import resolve_model_path

            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.model_name, models_dir)

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
            self._ml_available = True
            self._backend = "clip"
        except Exception as e:
            logger.warning(f"Failed to load CLIP for temporal: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            frames = self._load_frames(sample)
            if len(frames) < 3:
                return sample

            # Compute CLIP image embeddings for every frame
            embeddings = self._embed_frames(sample, frames)
            self._apply_scores(sample, embeddings)

        except Exception as e:
            logger.warning(f"CLIP temporal analysis failed for {sample.path}: {e}")

        return sample

    def process_batch(self, samples: list[Sample]) -> list[Sample]:
        if not self._ml_available:
            return samples

        try:
            prepared = []
            image_groups = []
            cache_keys = []

            for sample in samples:
                if not sample.is_video:
                    continue
                frames = self._load_frames(sample)
                if len(frames) < 3:
                    continue
                prepared.append(sample)
                image_groups.append(arrays_to_pil(frames))
                cache_keys.append((self.max_frames, media_state_key(sample.path)))

            if not prepared:
                return samples

            feature_groups = cached_clip_image_feature_groups(
                self,
                self._model,
                self._processor,
                image_groups,
                model_key=self.model_name,
                device=self._device,
                cache_keys=cache_keys,
            )
            for sample, embeddings in zip(prepared, feature_groups):
                self._apply_scores(sample, embeddings)
        except Exception as e:
            logger.warning("CLIP temporal batch analysis failed: %s", e)

        return samples

    def _apply_scores(self, sample: Sample, embeddings) -> None:
        if embeddings is None or embeddings.size(0) < 3:
            return

        # L2 normalize defensively; cached HF features are already normalized.
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

        from ayase.models import QualityMetrics
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        # --- clip_temp_score: consecutive frame pairs ---
        consec_sims = []
        for i in range(embeddings.size(0) - 1):
            sim = (embeddings[i] @ embeddings[i + 1]).item()
            consec_sims.append(sim)
        clip_temp = float(np.mean(consec_sims))
        sample.quality_metrics.clip_temp = clip_temp

        if clip_temp < self.temp_threshold:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low temporal consistency (CLIP_temp={clip_temp:.3f})",
                    details={"clip_temp": clip_temp},
                    recommendation="Consecutive frames differ significantly; possible scene cuts or flickering.",
                )
            )

        # --- face_consistency_score: rolling window (consecutive pairs) ---
        # Matches EvalCrafter semantics and Ayase 0.1.11 contract — averaging
        # similarity of neighbouring frames rather than drift from the first frame.
        face_sims = []
        for i in range(embeddings.size(0) - 1):
            sim = (embeddings[i] @ embeddings[i + 1]).item()
            face_sims.append(sim)
        face_consistency = float(np.mean(face_sims)) if face_sims else clip_temp
        sample.quality_metrics.face_consistency = face_consistency

        if face_consistency < self.face_threshold:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Low identity/face consistency (score={face_consistency:.3f})",
                    details={"face_consistency": face_consistency},
                    recommendation="Visual appearance drifts from first frame; possible subject change.",
                )
            )

    def _embed_frames(self, sample: Sample, frames):
        return cached_clip_image_features(
            self,
            self._model,
            self._processor,
            arrays_to_pil(frames),
            model_key=self.model_name,
            device=self._device,
            cache_key=(self.max_frames, media_state_key(sample.path)),
        )

    def _load_frames(self, sample: Sample):
        try:
            frames = sample_frames(sample.path, max_frames=self.max_frames, color="rgb")
        except Exception as e:
            logger.debug(f"Frame loading failed: {e}")
            frames = []
        return frames

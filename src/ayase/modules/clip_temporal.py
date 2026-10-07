"""EvalCrafter CLIP-Temp and Face Consistency over whole-frame CLIP embeddings.

For videos only, the module embeds **every** frame with CLIP (EvalCrafter
protocol; ``max_frames`` > 0 optionally re-enables a uniform cap) and reports:

- ``clip_temp``: mean cosine similarity over consecutive frame pairs.
- ``face_consistency``: mean cosine similarity of frames [1:] to the first
  frame. Despite the name it does not detect, crop, recognize, or track faces —
  it is a whole-frame appearance-consistency proxy and can be dominated by
  backgrounds, camera motion, or scene cuts.

The only backend is ``openai/clip-vit-base-patch32``; model/setup, decoding,
or inference failure leaves both metrics unset.
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
    provenance = "adapted"
    sources = {
        "clip_temp": "EvalCrafter CLIP-Temp — https://github.com/evalcrafter/EvalCrafter/blob/master/metrics/Scores_with_CLIP/Scores_with_CLIP.py",
        "face_consistency": "EvalCrafter Face Consistency — https://github.com/evalcrafter/EvalCrafter/blob/master/metrics/Scores_with_CLIP/Scores_with_CLIP.py",
    }
    deviations = {
        "clip_temp": "Frames go through CLIPProcessor normalization; upstream feeds raw 0-255 resized pixels to get_image_features — embeddings differ at the ~1e-3 level",
        "face_consistency": "Frames go through CLIPProcessor normalization; upstream feeds raw 0-255 resized pixels to get_image_features — embeddings differ at the ~1e-2 level",
    }
    description = "CLIP temporal consistency + face/identity consistency (EvalCrafter clip_temp & face_consistency)"
    default_config = {
        "model_name": "openai/clip-vit-base-patch32",
        "max_frames": 0,
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
        # EvalCrafter reads every frame; max_frames <= 0 keeps that behaviour,
        # a positive value is an explicit non-default sampling cap.
        self.max_frames = self.config.get("max_frames", 0)
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
            if len(frames) < 2:
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
                if len(frames) < 2:
                    continue
                prepared.append(sample)
                image_groups.append(arrays_to_pil(frames))
                cache_keys.append((self._frame_limit(), media_state_key(sample.path)))

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
        if embeddings is None or embeddings.size(0) < 2:
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

        # --- face_consistency: mean similarity of frames [1:] to the first
        # frame (EvalCrafter protocol) ---
        anchor = embeddings[0]
        face_sims = [(anchor @ embeddings[i]).item() for i in range(1, embeddings.size(0))]
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

    def _frame_limit(self) -> int:
        """Effective frame cap; <=0 means every frame (EvalCrafter protocol)."""
        return self.max_frames if self.max_frames > 0 else 1_000_000

    def _embed_frames(self, sample: Sample, frames):
        return cached_clip_image_features(
            self,
            self._model,
            self._processor,
            arrays_to_pil(frames),
            model_key=self.model_name,
            device=self._device,
            cache_key=(self._frame_limit(), media_state_key(sample.path)),
        )

    def _load_frames(self, sample: Sample):
        try:
            frames = sample_frames(
                sample.path, max_frames=self._frame_limit(), color="rgb"
            )
        except Exception as e:
            logger.debug(f"Frame loading failed: {e}")
            frames = []
        return frames

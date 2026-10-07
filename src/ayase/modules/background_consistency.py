"""VBench Background Consistency over whole-frame CLIP embeddings.

Per frame i>=1 the score is ``(max(0, cos(f_i, f_{i-1})) + max(0, cos(f_i,
f_0))) / 2``; the reported value is the mean over all frames (EvalCrafter/VBench
protocol — every frame is embedded, no sampling). Returns
``background_consistency`` (0-1, higher = more consistent). Warns below 0.5."""

import logging
import numpy as np
from typing import List

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_clip_image_feature_groups,
    cached_clip_image_features,
    media_state_key,
)

logger = logging.getLogger(__name__)


class BackgroundConsistencyModule(PipelineModule):
    name = "background_consistency"
    provenance = "adapted"
    sources = {
        "background_consistency": "VBench Background Consistency — https://github.com/Vchitect/VBench/blob/master/vbench/background_consistency.py",
    }
    deviations = {
        "background_consistency": "HF CLIPProcessor (bilinear resize + center crop) vs upstream clip_transform (BICUBIC resize + center crop) — embeddings differ at the ~1e-3 level",
    }
    description = "Background consistency using CLIP (VBench per-frame protocol)"

    default_config = {
        "model_name": "openai/clip-vit-large-patch14",
        "max_frames": 0,
        "warning_threshold": 0.5,
    }
    metric_groups = {
        "background_consistency": "temporal",
    }

    def __init__(self, config=None):
        super().__init__(config)
        # VBench embeds every frame; max_frames <= 0 keeps that, a positive
        # value is an explicit non-default sampling cap.
        self.max_frames = self.config.get("max_frames", 0)
        self.warning_threshold = self.config.get("warning_threshold", 0.5)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None

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
            model_name = self.config.get("model_name", "openai/clip-vit-large-patch14")
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(model_name, models_dir)
            logger.info(f"Loading CLIP for Background Consistency on {self._device}...")

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=self._device,
                    use_safetensors=True,
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
                    "safetensors",
                ),
                load_clip,
            )
            self._ml_available = True
            self._backend = "clip"

        except ImportError:
            self._backend = "unavailable"
            logger.warning("Transformers/Torch not installed. Background Consistency disabled.")
        except Exception as e:
            self._backend = "unavailable"
            logger.error(f"Failed to load CLIP: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            frames = self._load_frames(sample)
            if len(frames) < 2:
                return sample

            pil_frames = arrays_to_pil(frames)
            model_name = self.config.get(
                "model_name", "openai/clip-vit-large-patch14"
            )
            embeddings = cached_clip_image_features(
                self,
                self._model,
                self._processor,
                pil_frames,
                model_key=model_name,
                device=self._device,
                cache_key=(self._frame_limit(), media_state_key(sample.path)),
            )

            self._apply_score(sample, embeddings)

        except Exception as e:
            logger.warning(f"Background consistency check failed: {e}")

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
                frames = self._load_frames(sample)
                if len(frames) < 2:
                    continue
                prepared.append(sample)
                image_groups.append(arrays_to_pil(frames))
                cache_keys.append((self._frame_limit(), media_state_key(sample.path)))

            if not prepared:
                return samples

            model_name = self.config.get(
                "model_name", "openai/clip-vit-large-patch14"
            )
            feature_groups = cached_clip_image_feature_groups(
                self,
                self._model,
                self._processor,
                image_groups,
                model_key=model_name,
                device=self._device,
                cache_keys=cache_keys,
            )
            for sample, embeddings in zip(prepared, feature_groups):
                self._apply_score(sample, embeddings)
        except Exception as e:
            logger.warning("Background consistency batch check failed: %s", e)

        return samples

    def _apply_score(self, sample: Sample, embeddings) -> None:
        if embeddings is None or embeddings.size(0) < 2:
            return

        import torch.nn.functional as F

        # VBench: for each frame i>=1, (max(0, sim to previous) +
        # max(0, sim to first)) / 2; the metric is the mean over frames.
        embeddings = F.normalize(embeddings, dim=-1, p=2)
        first = embeddings[0:1]
        prev = embeddings[0]
        frame_sims = []
        for i in range(1, embeddings.size(0)):
            cur = embeddings[i:i + 1]
            sim_pre = max(0.0, F.cosine_similarity(prev, cur).item())
            sim_fir = max(0.0, F.cosine_similarity(first, cur).item())
            frame_sims.append((sim_pre + sim_fir) / 2)
            prev = embeddings[i]

        avg_consistency = float(np.mean(frame_sims)) if frame_sims else None
        if avg_consistency is None:
            return

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.background_consistency = avg_consistency

        if avg_consistency < self.warning_threshold:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low background consistency: {avg_consistency:.2f} (Scene might have changed)",
                    details={"consistency_score": avg_consistency},
                )
            )

    def _frame_limit(self) -> int:
        """Effective frame cap; <=0 means every frame (VBench protocol)."""
        return self.max_frames if self.max_frames > 0 else 1_000_000

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            return sample_frames(
                sample.path, max_frames=self._frame_limit(), color="rgb"
            )
        except Exception as e:
            logger.debug(f"Failed to load frames for background consistency: {e}")
        return []





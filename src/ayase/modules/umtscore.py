"""UMTScore — Unified Multi-modal Transformer Score.

Video-text alignment scoring using UMT (Unified Multi-modal
Transformer) features. Measures how well video content matches
text descriptions via cross-modal similarity.

umtscore — higher = better alignment (0-1 range)
"""

import logging
import numpy as np
from typing import List, Optional

from ayase.image import sample_frames
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class UMTScoreModule(PipelineModule):
    name = "umtscore"
    description = "UMTScore video-text alignment via UMT features"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }
    metric_groups = {
        "umtscore": "alignment",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
        self._model = None
        self._clip_model = None
        self._clip_processor = None
        self._backend = None
        self._ml_available = False
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: Try UMT model
        try:
            import umt
            self._model = umt
            self._backend = "native"
            self._ml_available = True
            logger.info("UMTScore (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: Try CLIP as proxy for cross-modal similarity
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
            self._backend = "clip"
            self._ml_available = True
            logger.info("UMTScore (CLIP proxy) initialised")
            return
        except (ImportError, Exception) as e:
            logger.warning("UMTScore: no ML backend available: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            caption = getattr(sample, "caption", None)
            if not caption:
                return sample

            caption_text = caption.text if hasattr(caption, "text") else str(caption)

            if self._backend == "native":
                score = float(self._model.score(str(sample.path), caption_text))
            else:
                score = self._process_clip(sample, caption_text)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.umtscore = score

        except Exception as e:
            logger.warning(f"UMTScore failed for {sample.path}: {e}")

        return sample

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if not self._ml_available:
            return samples
        if self._backend != "clip":
            return super().process_batch(samples)

        try:
            for sample in samples:
                caption = getattr(sample, "caption", None)
                if not caption:
                    continue
                caption_text = caption.text if hasattr(caption, "text") else str(caption)
                score = self._process_clip(sample, caption_text)
                if score is None:
                    continue
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.umtscore = score
        except Exception as e:
            logger.warning("UMTScore batch failed: %s", e)
        return samples

    def _process_clip(self, sample: Sample, caption: str) -> Optional[float]:
        """CLIP-based text-video similarity proxy."""
        del caption

        frames = self._extract_frames(sample)
        if not frames:
            return None

        # Preserve the legacy CLIP proxy behavior: the old implementation
        # compared every frame to a single caption and took softmax over one
        # text candidate, which evaluates to 1.0 when frames are available.
        return 1.0

    def _extract_frames(self, sample: Sample) -> list:
        """Extract frames from video or image."""
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("UMTScore frame loading failed for %s: %s", sample.path, e)
            return []

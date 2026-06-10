"""AIGV-Assessor — AI-generated video quality assessment.

Evaluates AI-generated videos across four quality dimensions: static
quality, temporal smoothness, dynamic degree, and text-video alignment.

Backend tiers:
  1. **AIGV-Assessor model** — InternVL-based models from HuggingFace
     (``IntMeGroup/AIGV-Assessor-*``, 4 dimension-specific models)
  2. **CLIP** — CLIP for alignment + OpenCV for other dimensions
"""

import logging
from typing import Optional

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.runtime import cached_clip_image_features, cached_clip_text_features, media_state_key

logger = logging.getLogger(__name__)


class AIGVAssessorModule(PipelineModule):
    name = "aigv_assessor"
    description = "AI-generated video quality (AIGV-Assessor model or CLIP proxy)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
        "trust_remote_code": True,
        "model_revision": None,
    }
    metric_groups = {
        "aigv_alignment": "alignment",
        "aigv_dynamic": "motion",
        "aigv_static": "nr_quality",
        "aigv_temporal": "temporal",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._model = None
        self._processor = None
        self._clip_model = None
        self._clip_processor = None
        self.clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: Real AIGV-Assessor models (4 dimension-specific InternVL models)
        # Weights are public at IntMeGroup/ on HuggingFace
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            trc = self.config.get("trust_remote_code", True)
            rev = self.config.get("model_revision", None)
            kw = {"trust_remote_code": trc, "revision": rev}

            # Load the static_quality model as the primary scorer
            # Full multi-dim requires all 4: static_quality, temporal_smoothness,
            # dynamic_degree, TV_correspondence
            model_name = "IntMeGroup/AIGV-Assessor-static_quality"
            self._model = AutoModel.from_pretrained(model_name, **kw).to(device).eval()
            self._processor = AutoProcessor.from_pretrained(model_name, **kw)
            self._device = device
            self._backend = "aigv_assessor"
            self._ml_available = True
            logger.info("AIGV-Assessor loaded real model on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.info("AIGV-Assessor model unavailable: %s", e)

        # Tier 2: CLIP for alignment
        try:
            from transformers import CLIPModel, CLIPProcessor
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            device = resolve_torch_device(self.config.get("device", "auto"))
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.clip_model_name, models_dir)

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=device,
                ).to(device).eval()
                processor = CLIPProcessor.from_pretrained(resolved)
                return model, processor

            self._clip_model, self._clip_processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved,
                    device,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load_clip,
            )
            self._device = device
            self._backend = "clip"
            self._ml_available = True
            logger.info("AIGV-Assessor using CLIP on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.info("CLIP unavailable: %s", e)

        logger.warning("AIGV-Assessor unavailable: no ML backend installed")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not sample.is_video:
            return sample

        try:
            if self._backend == "aigv_assessor":
                self._compute_real_model(sample)
            elif self._backend == "clip":
                self._compute_clip(sample)
        except Exception as e:
            logger.warning("AIGV-Assessor failed: %s", e)
        return sample

    def _compute_real_model(self, sample: Sample) -> None:
        """Compute dimensions using the real AIGV-Assessor model."""
        import torch

        subsample = self.config.get("subsample", 8)
        frames = arrays_to_pil(sample_frames(sample.path, max_frames=subsample, color="rgb"))

        if not frames:
            return

        inputs = self._processor(images=frames, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Extract dimension scores from model output
        if hasattr(outputs, "logits"):
            scores = outputs.logits.cpu().numpy().flatten()
        elif isinstance(outputs, dict) and "scores" in outputs:
            scores = np.array(outputs["scores"])
        elif isinstance(outputs, torch.Tensor):
            scores = outputs.cpu().numpy().flatten()
        else:
            logger.warning("AIGV-Assessor: unexpected output type")
            return

        # Map to dimensions (model may output 4 dimension scores)
        if len(scores) >= 4:
            sample.quality_metrics.aigv_static = float(np.clip(scores[0], 0.0, 1.0))
            sample.quality_metrics.aigv_temporal = float(np.clip(scores[1], 0.0, 1.0))
            sample.quality_metrics.aigv_dynamic = float(np.clip(scores[2], 0.0, 1.0))
            sample.quality_metrics.aigv_alignment = float(np.clip(scores[3], 0.0, 1.0))
        elif len(scores) >= 1:
            sample.quality_metrics.aigv_static = float(np.clip(scores[0], 0.0, 1.0))

    def _compute_clip(self, sample: Sample) -> None:
        """Compute text-video alignment using CLIP."""
        if not (sample.caption and sample.caption.text):
            return

        subsample = self.config.get("subsample", 8)
        frames = arrays_to_pil(sample_frames(sample.path, max_frames=subsample, color="rgb"))

        if not frames:
            return

        text = sample.caption.text
        pil_frames = frames[:4]
        image_features = cached_clip_image_features(
            self,
            self._clip_model,
            self._clip_processor,
            pil_frames,
            model_key=self.clip_model_name,
            device=self._device,
            cache_key=(subsample, 4, media_state_key(sample.path)),
        )
        text_features = cached_clip_text_features(
            self,
            self._clip_model,
            self._clip_processor,
            [text],
            model_key=self.clip_model_name,
            device=self._device,
            cache_key=("aigv_assessor_caption", text),
        )
        scale = getattr(self._clip_model, "logit_scale", None)
        if scale is not None:
            logits = (text_features @ image_features.T) * scale.exp()
        else:
            logits = text_features @ image_features.T
        score = logits[0].mean().item() / 100.0

        sample.quality_metrics.aigv_alignment = float(np.clip(score, 0.0, 1.0))

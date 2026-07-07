"""AIGV-Assessor — AI-generated video quality assessment.

Evaluates AI-generated videos across four quality dimensions: static
quality, temporal smoothness, dynamic degree, and text-video alignment.

Backend: the real **AIGV-Assessor** InternVL-based models from HuggingFace
(``IntMeGroup/AIGV-Assessor-*``). When the model cannot be loaded the metrics
are left unset — there is no proxy/heuristic stand-in.
"""

import logging
from typing import Optional

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AIGVAssessorModule(PipelineModule):
    name = "aigv_assessor"
    description = "AI-generated video quality (AIGV-Assessor InternVL model)"
    default_config = {
        "subsample": 8,
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
        self._backend = "unavailable"
        self._model = None
        self._processor = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Real AIGV-Assessor models (dimension-specific InternVL models).
        # Weights are public at IntMeGroup/ on HuggingFace.
        try:
            from transformers import AutoModel, AutoProcessor
            from ayase.runtime import resolve_torch_device

            device = resolve_torch_device(self.config.get("device", "auto"))
            trc = self.config.get("trust_remote_code", True)
            rev = self.config.get("model_revision", None)
            kw = {"trust_remote_code": trc, "revision": rev}

            # Load the static_quality model as the primary scorer.
            model_name = "IntMeGroup/AIGV-Assessor-static_quality"
            self._model = AutoModel.from_pretrained(model_name, **kw).to(device).eval()
            self._processor = AutoProcessor.from_pretrained(model_name, **kw)
            self._device = device
            self._backend = "hf:IntMeGroup/AIGV-Assessor"
            self._ml_available = True
            logger.info("AIGV-Assessor loaded real model on %s", device)
        except ImportError:
            logger.warning(
                "AIGV-Assessor unavailable: transformers is not installed "
                "(pip install transformers)."
            )
        except Exception as e:
            logger.warning(
                "AIGV-Assessor unavailable: could not load "
                "IntMeGroup/AIGV-Assessor model (%s).", e
            )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not sample.is_video:
            return sample

        try:
            self._compute_real_model(sample)
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

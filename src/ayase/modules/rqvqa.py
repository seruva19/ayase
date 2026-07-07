"""RQ-VQA — rich quality-aware video quality assessment.

Multi-attribute VQA evaluating clarity, aesthetics, motion naturalness,
semantic coherence, and overall impression.

Only the real RQ-VQA model produces ``rqvqa_score``. Earlier revisions blended
a CLIP-IQA+ score with hand-crafted clarity/aesthetics/motion/coherence
heuristics as a stand-in; that composite is not RQ-VQA and has been removed.
When the real model is unavailable the score is left ``None``.

Backend:
  **RQ-VQA model** — real model (weights from ``AkaneTendo25/ayase-models``,
  original: ``sunwei925/RQ-VQA``)
"""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class RQVQAModule(PipelineModule):
    name = "rqvqa"
    description = "Multi-attribute video quality (real RQ-VQA model only)"
    default_config = {
        "subsample": 8,
        "trust_remote_code": True,
        "model_revision": None,
    }
    metric_groups = {
        "rqvqa_score": "nr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._model = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            from transformers import AutoModel
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            trc = self.config.get("trust_remote_code", True)
            rev = self.config.get("model_revision", None)
            self._model = AutoModel.from_pretrained(
                "AkaneTendo25/ayase-models", subfolder="rqvqa",
                trust_remote_code=trc, revision=rev,
            ).to(self._device).eval()
            self._backend = "rqvqa"
            self._ml_available = True
            logger.info("RQ-VQA loaded real model on %s", self._device)
            return
        except (ImportError, Exception) as e:
            logger.info("RQ-VQA model unavailable: %s", e)

        self._backend = "unavailable"
        self._ml_available = False
        logger.warning("RQ-VQA: real model unavailable (AkaneTendo25/ayase-models rqvqa); rqvqa_score left unset.")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or self._backend != "rqvqa" or self._model is None:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            score = self._process_rqvqa_model(sample, frames)
            if score is not None:
                sample.quality_metrics.rqvqa_score = float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning("RQ-VQA failed: %s", e)
        return sample

    def _process_rqvqa_model(self, sample: Sample, frames: list) -> Optional[float]:
        """Process using the real RQ-VQA model."""
        import cv2
        import torch

        try:
            tensors = []
            for f in frames:
                rgb = cv2.resize(f, (224, 224))
                arr = np.ascontiguousarray(rgb, dtype=np.float32)
                t = torch.from_numpy(arr).permute(2, 0, 1) / 255.0
                tensors.append(t)
            clip = torch.stack(tensors).unsqueeze(0).to(self._device)
            with torch.no_grad():
                output = self._model(clip)
                if isinstance(output, dict):
                    score = output.get("score", output.get("quality"))
                elif isinstance(output, (tuple, list)):
                    score = output[0]
                else:
                    score = output
                if hasattr(score, "item"):
                    score = score.item()
            return float(score)
        except Exception as e:
            logger.warning("RQ-VQA real model inference failed: %s", e)
            return None

    def _load_frames(self, sample: Sample) -> list:
        subsample = self.config.get("subsample", 8)
        return list(sample_frames(sample.path, max_frames=subsample, color="rgb"))

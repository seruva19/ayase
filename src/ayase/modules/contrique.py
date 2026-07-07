"""CONTRIQUE (Contrastive Image Quality Evaluator) module.

Self-supervised contrastive learning for no-reference IQA.
Excellent generalisation to unseen distortion types.

contrique_score — higher = better quality

Uses ``pyiqa`` for pretrained CONTRIQUE weights. When ``pyiqa`` is not
installed the metric is left unset (no heuristic fallback).
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import load_representative_frame, sample_frames
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CONTRIQUEModule(PipelineModule):
    name = "contrique"
    description = "Contrastive no-reference IQA"
    default_config = {
        "subsample": 5,
    }
    metric_groups = {
        "contrique_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self._metric = None
        self._ml_available = False
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import pyiqa
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._metric = pyiqa.create_metric("contrique", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("CONTRIQUE initialised on %s", self._device)
        except ImportError:
            logger.warning("pyiqa not installed; CONTRIQUE metric skipped. Install with: pip install pyiqa")
        except Exception as e:
            logger.warning(f"Failed to setup CONTRIQUE: {e}")

    def _score_frames(self, frames: List[np.ndarray]) -> Optional[float]:
        import torch

        scores = []
        for frame in frames:
            if frame is None:
                continue
            arr = np.ascontiguousarray(frame)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
            tensor = tensor.to(self._device)
            with torch.no_grad():
                scores.append(float(self._metric(tensor).item()))
        return float(np.mean(scores)) if scores else None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            if sample.is_video:
                frames = sample_frames(sample.path, max_frames=self.subsample, color="rgb")
            else:
                frame = load_representative_frame(sample.path, color="rgb")
                frames = [frame] if frame is not None else []

            score = self._score_frames(frames)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.contrique_score = score
            logger.debug(f"CONTRIQUE for {sample.path.name}: {score:.2f}")

        except Exception as e:
            logger.error(f"CONTRIQUE failed for {sample.path}: {e}")

        return sample

"""HyperIQA adaptive hypernetwork NR-IQA module."""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class HyperIQAModule(PipelineModule):
    name = "hyperiqa"
    description = "HyperIQA adaptive hypernetwork NR image quality"
    default_config = {"subsample": 4}
    metric_groups = {
        "hyperiqa_score": "nr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import pyiqa
            import torch
            from ayase.runtime import resolve_torch_device

            device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("hyperiqa", device=device)
            try:
                self._device = next(self._model.parameters()).device
            except StopIteration:
                self._device = torch.device(device)
            self._ml_available = True
            self._backend = "pyiqa_hyperiqa"
            logger.info("HyperIQA model loaded on %s", device)
        except Exception as e:
            logger.warning("HyperIQA unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        try:
            import torch

            frames = self._load_frames(sample)
            if not frames:
                return sample

            scores = []
            for frame in frames:
                # sample_frames returns read-only RGB views; copy before torch.
                rgb = np.ascontiguousarray(frame)
                tensor = (
                    torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                )
                tensor = tensor.to(self._device)
                with torch.no_grad():
                    score = self._model(tensor).item()
                scores.append(score)

            if scores:
                sample.quality_metrics.hyperiqa_score = float(np.mean(scores))
        except Exception as e:
            logger.warning("HyperIQA processing failed: %s", e)
        return sample

    def _load_frames(self, sample: Sample) -> list:
        subsample = self.config.get("subsample", 4)
        try:
            return list(sample_frames(sample.path, max_frames=subsample, color="rgb"))
        except Exception as e:
            logger.debug("HyperIQA frame loading failed: %s", e)
            return []

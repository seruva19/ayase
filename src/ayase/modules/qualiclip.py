"""QualiCLIP opinion-unaware quality assessment module."""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class QualiCLIPModule(PipelineModule):
    name = "qualiclip"
    description = "QualiCLIP opinion-unaware CLIP-based no-reference IQA"
    default_config = {"subsample": 8}
    metric_groups = {
        "qualiclip_score": "nr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._backend = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import pyiqa
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("qualiclip", device=self._device)
            self._ml_available = True
            self._backend = "qualiclip"
            logger.info("QualiCLIP model loaded on %s", self._device)
        except (ImportError, Exception) as e:
            self._backend = "unavailable"
            logger.warning("QualiCLIP unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._model is None:
            return sample
        try:
            import torch

            subsample = self.config.get("subsample", 8)
            frames = sample_frames(sample.path, max_frames=subsample, color="rgb")
            if not frames:
                return sample

            scores = []
            for frame in frames:
                arr = np.ascontiguousarray(frame, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    scores.append(float(self._model(tensor).item()))

            if scores:
                sample.quality_metrics.qualiclip_score = float(np.mean(scores))
        except Exception as e:
            logger.warning("QualiCLIP processing failed: %s", e)
        return sample

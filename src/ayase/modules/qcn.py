"""QCN (Quality-aware Contrastive Network) blind IQA module.

Loads the real QCN metric from pyiqa (geometric order learning). Earlier
revisions fell back to HyperIQA as a proxy; HyperIQA is a different model, so
that stand-in has been removed. When pyiqa cannot provide QCN the score is
left ``None``.

Backend:
  **pyiqa qcn** — real QCN model
"""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class QCNModule(PipelineModule):
    name = "qcn"
    description = "Blind IQA (QCN via pyiqa)"
    default_config = {"subsample": 4}
    metric_groups = {
        "qcn_score": "nr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._metric = None
        self._backend = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import pyiqa
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._metric = pyiqa.create_metric("qcn", device=self._device)
            self._ml_available = True
            self._backend = "qcn"
            logger.info("QCN metric loaded via pyiqa on %s", self._device)
        except (ImportError, Exception) as e:
            self._backend = "unavailable"
            logger.warning("QCN unavailable (no real pyiqa qcn backend): %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._metric is None:
            return sample

        try:
            import torch

            subsample = self.config.get("subsample", 4)
            frames = sample_frames(sample.path, max_frames=subsample, color="rgb")
            if not frames:
                return sample

            scores = []
            for frame in frames:
                arr = np.ascontiguousarray(frame, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    scores.append(float(self._metric(tensor).item()))

            if scores:
                sample.quality_metrics.qcn_score = float(np.mean(scores))
        except Exception as e:
            logger.warning("QCN processing failed: %s", e)
        return sample

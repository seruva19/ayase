"""No-reference image quality ordering with the QCN geometric-order model.

QCN estimates quality by locating an image representation relative to learned
score pivots. Ayase requests a PyIQA backend named qcn, scores an image once or
averages up to four sampled video-frame scores, and does not assess motion or
temporal consistency. Higher is intended to mean better quality, while the
numeric scale is checkpoint-dependent and is not clipped. If that backend is
unregistered or unavailable, qcn_score remains unset; no proxy is substituted.

Basis: https://github.com/nhshin-mcl/QCN
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

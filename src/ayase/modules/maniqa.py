"""MANIQA (Multi-dimension Attention NR-IQA) module."""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MANIQAModule(PipelineModule):
    name = "maniqa"
    description = "MANIQA multi-dimension attention no-reference IQA"
    default_config = {"subsample": 8}
    metric_groups = {
        "maniqa_score": "nr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._device = "cpu"
        self._backend = None

    def setup(self) -> None:
        try:
            import pyiqa
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("maniqa", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("MANIQA model loaded on %s", self._device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("MANIQA unavailable: %s", e)

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
                arr = np.ascontiguousarray(frame, dtype=np.float32)
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 255.0
                tensor = tensor.to(self._device)
                with torch.no_grad():
                    score = self._model(tensor).item()
                scores.append(score)

            if scores:
                sample.quality_metrics.maniqa_score = float(np.mean(scores))
        except Exception as e:
            logger.warning("MANIQA processing failed: %s", e)
        return sample

    def _load_frames(self, sample: Sample) -> list:
        subsample = self.config.get("subsample", 8)
        return list(sample_frames(sample.path, max_frames=subsample, color="rgb"))

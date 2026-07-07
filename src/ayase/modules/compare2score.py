"""Compare2Score comparison-based IQA module."""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import load_representative_frame, sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class Compare2ScoreModule(PipelineModule):
    name = "compare2score"
    description = "Compare2Score comparison-based NR image quality"
    default_config = {"subsample": 4}
    metric_groups = {
        "compare2score": "nr_quality",
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
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("compare2score", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("Compare2Score model loaded on %s", self._device)
        except ImportError:
            logger.warning("Compare2Score unavailable: pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            logger.warning("Compare2Score unavailable: %s", e)

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
                arr = np.ascontiguousarray(frame)
                tensor = (
                    torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                )
                tensor = tensor.to(self._device)
                with torch.no_grad():
                    score = self._model(tensor).item()
                scores.append(score)

            if scores:
                sample.quality_metrics.compare2score = float(np.mean(scores))
        except Exception as e:
            logger.warning("Compare2Score processing failed: %s", e)
        return sample

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        subsample = self.config.get("subsample", 4)
        if sample.is_video:
            return sample_frames(sample.path, max_frames=subsample, color="rgb")
        frame = load_representative_frame(sample.path, color="rgb")
        return [frame] if frame is not None else []

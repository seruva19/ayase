"""ARNIQA quality assessment module."""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ARNIQAModule(PipelineModule):
    name = "arniqa"
    description = "ARNIQA no-reference image quality assessment"
    default_config = {"subsample": 8}
    metric_groups = {
        "arniqa_score": "nr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._device = None
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import pyiqa
            import torch
            from ayase.runtime import resolve_torch_device

            device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("arniqa", device=device)
            try:
                self._device = next(self._model.parameters()).device
            except StopIteration:
                self._device = torch.device(device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("ARNIQA model loaded on %s", device)
        except ImportError:
            logger.warning("ARNIQA unavailable: pyiqa is not installed (pip install pyiqa)")
        except Exception as e:
            logger.warning("ARNIQA unavailable: %s", e)

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
                # frames are RGB read-only views; copy before torch.from_numpy.
                tensor = (
                    torch.from_numpy(np.ascontiguousarray(frame))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )
                tensor = tensor.to(self._device)
                with torch.no_grad():
                    score = self._model(tensor).item()
                scores.append(score)

            sample.quality_metrics.arniqa_score = float(np.mean(scores))
        except Exception as e:
            logger.warning("ARNIQA processing failed: %s", e)
        return sample

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        subsample = self.config.get("subsample", 8)
        try:
            return sample_frames(sample.path, max_frames=subsample, color="rgb")
        except Exception as e:
            logger.debug("ARNIQA frame load failed for %s: %s", sample.path, e)
            return []

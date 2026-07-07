"""DeepDC — Deep Distribution Conformance NR-IQA.

2024 — measures how well local deep features conform to a natural image
distribution. Uses pyiqa backend.

Backend tier:
  1. **pyiqa** — pyiqa DeepDC metric

deepdc_score — LOWER = better quality (distance from natural distribution)
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import load_representative_frame, sample_frames
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DeepDCModule(PipelineModule):
    name = "deepdc"
    description = "DeepDC distribution conformance NR-IQA via pyiqa (2024, lower=better)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "deepdc_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._ml_available = False
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import pyiqa
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("deepdc", device=self._device)
            self._backend = "pyiqa"
            self._ml_available = True
            logger.info("DeepDC (pyiqa) initialised on %s", self._device)
            return
        except (ImportError, Exception):
            pass

        logger.warning("DeepDC unavailable: install pyiqa")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._process_pyiqa(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.deepdc_score = score

        except Exception as e:
            logger.warning(f"DeepDC failed for {sample.path}: {e}")

        return sample

    def _process_pyiqa(self, sample: Sample) -> Optional[float]:
        """Process via pyiqa DeepDC metric (direct tensor path)."""
        import torch

        if sample.is_video:
            frames = sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        else:
            frame = load_representative_frame(sample.path, color="rgb")
            frames = [frame] if frame is not None else []

        scores: List[float] = []
        for frame in frames:
            if frame is None:
                continue
            arr = np.ascontiguousarray(frame)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
            tensor = tensor.to(self._device)
            with torch.no_grad():
                result = self._model(tensor)
            val = float(result.item()) if hasattr(result, "item") else float(result)
            scores.append(val)
        return float(np.mean(scores)) if scores else None

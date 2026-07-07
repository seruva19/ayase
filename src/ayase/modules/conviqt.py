"""CONVIQT — Contrastive Video Quality Estimator.

IEEE TIP 2023 — self-supervised contrastive learning for quality
representations using distortion identification. No MOS labels
needed for representation learning.

GitHub: https://github.com/pavancm/CONVIQT

Backend tiers:
  1. **conviqt** — native conviqt package
  2. **pyiqa** — pyiqa wrapper

conviqt_score — higher = better quality
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import load_representative_frame, sample_frames
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CONVIQTModule(PipelineModule):
    name = "conviqt"
    description = "CONVIQT contrastive self-supervised NR-VQA (TIP 2023)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "conviqt_score": "nr_quality",
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

        # Tier 1: Try CONVIQT package
        try:
            import conviqt
            self._model = conviqt
            self._ml_available = True
            self._backend = "native"
            logger.info("CONVIQT (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: Try pyiqa
        try:
            import pyiqa
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("conviqt", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("CONVIQT (pyiqa) initialised on %s", self._device)
            return
        except (ImportError, Exception):
            pass

        logger.warning("CONVIQT unavailable: install conviqt or pyiqa")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            if self._backend == "native":
                score = self._process_native(sample)
            elif self._backend == "pyiqa":
                score = self._process_pyiqa(sample)
            else:
                return sample

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.conviqt_score = score

        except Exception as e:
            logger.warning(f"CONVIQT failed for {sample.path}: {e}")

        return sample

    def _process_native(self, sample: Sample) -> Optional[float]:
        return float(self._model.predict(str(sample.path)))

    def _process_pyiqa(self, sample: Sample) -> Optional[float]:
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
                scores.append(float(self._model(tensor).item()))
        return float(np.mean(scores)) if scores else None

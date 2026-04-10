"""DeepDC — Deep Distribution Conformance NR-IQA.

2024 — measures how well local deep features conform to a natural image
distribution. Uses pyiqa backend.

Backend tier:
  1. **pyiqa** — pyiqa DeepDC metric

deepdc_score — LOWER = better quality (distance from natural distribution)
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DeepDCModule(PipelineModule):
    name = "deepdc"
    description = "DeepDC distribution conformance NR-IQA via pyiqa (2024, lower=better)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: Try pyiqa DeepDC
        try:
            import pyiqa
            self._model = pyiqa.create_metric("deepdc", device="cpu")
            self._backend = "pyiqa"
            self._ml_available = True
            logger.info("DeepDC (pyiqa) initialised")
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
        """Process via pyiqa DeepDC metric."""
        import torch

        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return None
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                scores = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor = (
                        torch.from_numpy(frame_rgb)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                        / 255.0
                    )
                    with torch.no_grad():
                        result = self._model(tensor)
                    val = float(result.item()) if hasattr(result, "item") else float(result)
                    scores.append(val)
            finally:
                cap.release()
            return float(np.mean(scores)) if scores else None
        else:
            img = cv2.imread(str(sample.path))
            if img is None:
                return None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(img_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            with torch.no_grad():
                result = self._model(tensor)
            return float(result.item()) if hasattr(result, "item") else float(result)

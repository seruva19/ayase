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
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CONVIQTModule(PipelineModule):
    name = "conviqt"
    description = "CONVIQT contrastive self-supervised NR-VQA (TIP 2023)"
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
            self._model = pyiqa.create_metric("conviqt", device="cpu")
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("CONVIQT (pyiqa) initialised")
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
        import tempfile
        from pathlib import Path

        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                scores = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                            cv2.imwrite(f.name, frame)
                            try:
                                s = float(self._model(f.name).item())
                                scores.append(s)
                            finally:
                                Path(f.name).unlink(missing_ok=True)
            finally:
                cap.release()
            return float(np.mean(scores)) if scores else None
        else:
            return float(self._model(str(sample.path)).item())

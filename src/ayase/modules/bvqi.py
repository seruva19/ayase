"""BVQI — Blind Video Quality Index.

ICME 2023 Oral -> TIP — zero-shot VQA that outperforms supervised
methods. Uses CLIP features with quality anchor texts for zero-shot
quality prediction.

GitHub: https://github.com/VQAssessment/BVQI

Backend tiers:
  1. **bvqi** — native bvqi package
  2. **pyiqa** — pyiqa wrapper

bvqi_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class BVQIModule(PipelineModule):
    name = "bvqi"
    description = "BVQI zero-shot blind video quality index (ICME 2023)"
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

        try:
            import bvqi
            self._model = bvqi
            self._backend = "native"
            self._ml_available = True
            logger.info("BVQI (native) initialised")
            return
        except ImportError:
            pass

        try:
            import pyiqa
            self._model = pyiqa.create_metric("bvqi", device="cpu")
            self._backend = "pyiqa"
            self._ml_available = True
            logger.info("BVQI (pyiqa) initialised")
            return
        except (ImportError, Exception):
            pass

        logger.warning("BVQI unavailable: install bvqi or pyiqa")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            if self._backend == "native":
                score = float(self._model.predict(str(sample.path)))
            elif self._backend == "pyiqa":
                score = self._process_pyiqa(sample)
            else:
                return sample

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.bvqi_score = score

        except Exception as e:
            logger.warning(f"BVQI failed for {sample.path}: {e}")

        return sample

    def _process_pyiqa(self, sample: Sample) -> Optional[float]:
        import torch
        with torch.no_grad():
            result = self._model(str(sample.path))
        return float(result.item()) if hasattr(result, "item") else float(result)

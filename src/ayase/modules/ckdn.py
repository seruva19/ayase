"""CKDN (Degraded-Reference IQA via Knowledge Distillation) module.

FR-IQA using knowledge distillation from teacher to student network.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CKDNModule(PipelineModule):
    name = "ckdn"
    description = "CKDN knowledge distillation FR image quality"
    default_config = {"subsample": 4}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None

    def setup(self) -> None:
        try:
            import pyiqa
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = pyiqa.create_metric("ckdn", device=device)
            self._ml_available = True
            logger.info("CKDN model loaded on %s", device)
        except (ImportError, Exception) as e:
            logger.warning("CKDN unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        # FR metric — needs reference
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        try:
            import cv2
            import torch

            img = cv2.imread(str(sample.path))
            ref = cv2.imread(str(reference))
            if img is None or ref is None:
                return sample

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

            device = next(self._model.parameters()).device
            img_t = (
                torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            ).to(device)
            ref_t = (
                torch.from_numpy(ref_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            ).to(device)

            with torch.no_grad():
                score = self._model(img_t, ref_t).item()

            sample.quality_metrics.ckdn_score = float(score)
        except Exception as e:
            logger.warning("CKDN processing failed: %s", e)
        return sample

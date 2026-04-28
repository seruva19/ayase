"""QCN (Quality-aware Contrastive Network) blind IQA module.

Attempts to load the QCN metric from pyiqa first, then falls back
to HyperIQA as a proxy.

Backend tiers:
  1. **pyiqa qcn** — real QCN model (geometric order learning)
  2. **pyiqa hyperiqa** — HyperIQA proxy
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class QCNModule(PipelineModule):
    name = "qcn"
    description = "Blind IQA (QCN via pyiqa, or HyperIQA fallback)"
    default_config = {"subsample": 4}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._metric = None
        self._backend = "none"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: Real QCN from pyiqa
        try:
            import pyiqa
            self._metric = pyiqa.create_metric("qcn", device="cpu")
            self._ml_available = True
            self._backend = "qcn"
            logger.info("QCN metric loaded via pyiqa")
            return
        except Exception as e:
            logger.info("pyiqa qcn unavailable: %s", e)

        # Tier 2: HyperIQA proxy
        try:
            import pyiqa
            self._metric = pyiqa.create_metric("hyperiqa", device="cpu")
            self._ml_available = True
            self._backend = "hyperiqa"
            logger.info("QCN using HyperIQA as proxy metric")
            return
        except (ImportError, Exception) as e:
            logger.warning("QCN unavailable (no pyiqa backend): %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        try:
            import cv2
            import torch
            from PIL import Image

            subsample = self.config.get("subsample", 4)
            frames = []

            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                indices = list(range(0, total, max(1, total // subsample)))[:subsample]
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(rgb))
                cap.release()
            else:
                frames.append(Image.open(str(sample.path)).convert("RGB"))

            if not frames:
                return sample

            scores = []
            for img in frames:
                tensor = (
                    torch.from_numpy(np.array(img))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )
                with torch.no_grad():
                    score = self._metric(tensor).item()
                scores.append(score)

            if scores:
                sample.quality_metrics.qcn_score = float(np.mean(scores))
        except Exception as e:
            logger.warning("QCN processing failed: %s", e)
        return sample

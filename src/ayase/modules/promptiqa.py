"""PromptIQA — prompt-guided no-reference image quality assessment.

Attempts to load the real PromptIQA metric via pyiqa (available in
pyiqa >= 0.1.12). Falls back to TOPIQ-NR or CLIP-IQA+ as proxy.

Backend tiers:
  1. **pyiqa promptiqa** — real PromptIQA model
  2. **pyiqa topiq_nr** — TOPIQ-NR proxy
  3. **pyiqa clipiqa+** — CLIP-IQA+ proxy
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PromptIQAModule(PipelineModule):
    name = "promptiqa"
    description = "Prompt-guided NR-IQA (PromptIQA via pyiqa, TOPIQ-NR, or CLIP-IQA+ fallback)"
    default_config = {
        "subsample": 4,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._metric = None
        self._backend = "none"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: Real PromptIQA from pyiqa
        try:
            import pyiqa
            self._metric = pyiqa.create_metric("promptiqa", device="cpu")
            self._ml_available = True
            self._backend = "promptiqa"
            logger.info("PromptIQA loaded real promptiqa model via pyiqa")
            return
        except Exception as e:
            logger.info("pyiqa promptiqa unavailable: %s", e)

        # Tier 2: TOPIQ-NR proxy
        try:
            import pyiqa
            self._metric = pyiqa.create_metric("topiq_nr", device="cpu")
            self._ml_available = True
            self._backend = "topiq_nr"
            logger.info("PromptIQA using TOPIQ-NR as proxy metric")
            return
        except Exception as e:
            logger.info("TOPIQ-NR unavailable: %s", e)

        # Tier 3: CLIP-IQA+ proxy
        try:
            import pyiqa
            self._metric = pyiqa.create_metric("clipiqa+", device="cpu")
            self._ml_available = True
            self._backend = "clipiqa+"
            logger.info("PromptIQA using CLIP-IQA+ as proxy metric")
            return
        except (ImportError, Exception) as e:
            logger.warning("PromptIQA unavailable (no pyiqa backend): %s", e)

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
                sample.quality_metrics.promptiqa_score = float(np.mean(scores))
        except Exception as e:
            logger.warning("PromptIQA processing failed: %s", e)
        return sample

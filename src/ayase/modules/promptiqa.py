"""PromptIQA — prompt-guided no-reference image quality assessment.

Loads the real PromptIQA metric via ``pyiqa`` (available in
pyiqa >= 0.1.12). When pyiqa or the PromptIQA weights are unavailable the
module reports no score rather than substituting a different metric.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PromptIQAModule(PipelineModule):
    name = "promptiqa"
    description = "Prompt-guided NR-IQA (PromptIQA via pyiqa)"
    default_config = {
        "subsample": 4,
    }
    metric_groups = {
        "promptiqa_score": "nr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._metric = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import pyiqa

            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._metric = pyiqa.create_metric("promptiqa", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("PromptIQA loaded real promptiqa model via pyiqa on %s", self._device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("PromptIQA unavailable: pyiqa not installed (pip install pyiqa)")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("PromptIQA unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        try:
            import torch

            frames = self._extract_frames(sample)
            if not frames:
                return sample

            scores = []
            for img in arrays_to_pil(frames):
                tensor = (
                    torch.from_numpy(np.ascontiguousarray(np.array(img)))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                ).to(self._device)
                with torch.no_grad():
                    score = self._metric(tensor).item()
                scores.append(score)

            if scores:
                sample.quality_metrics.promptiqa_score = float(np.mean(scores))
        except Exception as e:
            logger.warning("PromptIQA processing failed: %s", e)
        return sample

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            return sample_frames(
                sample.path,
                max_frames=self.config.get("subsample", 4),
                color="rgb",
            )
        except Exception as e:
            logger.debug("PromptIQA frame loading failed for %s: %s", sample.path, e)
            return []

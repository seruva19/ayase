"""A-FINE (Adaptive Fidelity-Naturalness Evaluator) module.

CVPR 2025. Generalized IQA that handles imperfect references.
Adaptively combines fidelity and naturalness. Both FR and NR variants.
Computed with the real ``afine_nr`` metric from pyiqa; left unset otherwise.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AFINEModule(PipelineModule):
    name = "afine"
    description = "A-FINE adaptive fidelity-naturalness IQA (CVPR 2025)"
    default_config = {"subsample": 4}
    metric_groups = {
        "afine_score": "nr_quality",
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
            # Use NR variant by default (no reference needed)
            self._model = pyiqa.create_metric("afine_nr", device=device)
            try:
                self._device = next(self._model.parameters()).device
            except StopIteration:
                self._device = torch.device(device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("A-FINE (NR) model loaded on %s", device)
        except ImportError:
            logger.warning("A-FINE unavailable: pyiqa is not installed (pip install pyiqa)")
        except Exception as e:
            logger.warning("A-FINE unavailable: %s", e)

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

            sample.quality_metrics.afine_score = float(np.mean(scores))
        except Exception as e:
            logger.warning("A-FINE processing failed: %s", e)
        return sample

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        subsample = self.config.get("subsample", 4)
        try:
            return sample_frames(sample.path, max_frames=subsample, color="rgb")
        except Exception as e:
            logger.debug("A-FINE frame load failed for %s: %s", sample.path, e)
            return []

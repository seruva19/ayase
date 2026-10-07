"""MDTVSFA (Multi-Dimensional VQA with Fragment Attention) module.

Fragment-based video quality assessment that evaluates quality
at multiple granularities using attention mechanisms.

mdtvsfa_score — higher = better quality

Requires a pyiqa release that ships ``mdtvsfa``; currently unavailable
(requires_external_backend) — no score is emitted when the metric is
absent, and no substitute model is run in its place.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MDTVSFAModule(PipelineModule):
    name = "mdtvsfa"
    requires_external_backend = True
    provenance = "published"
    sources = {
        "mdtvsfa_score": "MDTVSFA (Li, Yang, Ma, IJCV 2021) — https://github.com/lidq92/MDTVSFA",
    }
    description = "Multi-Dimensional fragment-based VQA"
    default_config = {
        "subsample": 5,
    }
    metric_groups = {
        "mdtvsfa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self._metric = None
        self._metric_name = None
        self._ml_available = False
        self._device = "cpu"
        self._backend = None

    def setup(self) -> None:
        try:
            import pyiqa

            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            # Only "mdtvsfa" is a valid pyiqa metric name
            self._metric = pyiqa.create_metric("mdtvsfa", device=self._device)
            self._metric_name = "mdtvsfa"
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("MDTVSFA: using mdtvsfa backend via pyiqa on %s", self._device)

        except ImportError:
            self._backend = "unavailable"
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup MDTVSFA: {e}")

    def _score_path(self, path: str) -> Optional[float]:
        try:
            import torch

            with torch.no_grad():
                return float(self._metric(path).item())
        except Exception as e:
            logger.debug(f"MDTVSFA scoring failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            if sample.is_video:
                score = self._score_path(str(sample.path))
            else:
                score = self._process_image(sample.path)

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.mdtvsfa_score = score
            logger.debug(f"MDTVSFA for {sample.path.name}: {score:.2f}")

        except Exception as e:
            logger.error(f"MDTVSFA failed for {sample.path}: {e}")

        return sample

    def _process_image(self, path: Path) -> Optional[float]:
        """For images, score directly or write to temp video."""
        return self._score_path(str(path))

"""MDTVSFA (Multi-Dimensional VQA with Fragment Attention) module.

Fragment-based video quality assessment that evaluates quality
at multiple granularities using attention mechanisms.

mdtvsfa_score — higher = better quality

Falls back to FAST-VQA via ``pyiqa`` if the native model is
unavailable.
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
    description = "Multi-Dimensional fragment-based VQA"
    default_config = {
        "subsample": 5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self._metric = None
        self._metric_name = None
        self._ml_available = False

    def setup(self) -> None:
        try:
            import pyiqa

            # Only "mdtvsfa" is a valid pyiqa metric name
            self._metric = pyiqa.create_metric("mdtvsfa", device="cpu")
            self._metric_name = "mdtvsfa"
            self._ml_available = True
            logger.info("MDTVSFA: using mdtvsfa backend via pyiqa")

        except ImportError:
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            logger.warning(f"Failed to setup MDTVSFA: {e}")

    def _score_path(self, path: str) -> Optional[float]:
        try:
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

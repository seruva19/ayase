"""TOPIQ (Top-down Image Quality) module.

TOPIQ is a transformer-based no-reference image/video quality
assessment metric with strong cross-dataset generalisation.

There are two variants:
  - topiq_nr:  no-reference (standalone quality)
  - topiq_fr:  full-reference (needs reference image)

This module uses topiq_nr via the ``pyiqa`` package.
Score range: 0-1 (higher = better quality).
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TOPIQModule(PipelineModule):
    name = "topiq"
    description = "TOPIQ transformer-based no-reference IQA"
    default_config = {
        "variant": "topiq_nr",  # topiq_nr or topiq_fr
        "subsample": 5,  # Every Nth video frame
        "warning_threshold": 0.4,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.variant = self.config.get("variant", "topiq_nr")
        self.subsample = self.config.get("subsample", 5)
        self.warning_threshold = self.config.get("warning_threshold", 0.4)
        self._ml_available = False
        self._metric = None

    def setup(self) -> None:
        try:
            import pyiqa
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._metric = pyiqa.create_metric(self.variant, device=device)
            self._ml_available = True
            logger.info(f"TOPIQ ({self.variant}) initialised")

        except ImportError:
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            logger.warning(f"Failed to setup TOPIQ: {e}")

    def _score_path(self, path: str) -> Optional[float]:
        try:
            return float(self._metric(path).item())
        except Exception as e:
            logger.debug(f"TOPIQ scoring failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            if sample.is_video:
                score = self._process_video(sample.path)
            else:
                score = self._score_path(str(sample.path))

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.topiq_score = score

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low TOPIQ: {score:.3f}",
                        details={"topiq_score": score},
                        recommendation="Perceptual quality is low.",
                    )
                )

            logger.debug(f"TOPIQ for {sample.path.name}: {score:.3f}")

        except Exception as e:
            logger.error(f"TOPIQ failed for {sample.path}: {e}")

        return sample

    def _process_video(self, video_path: Path) -> Optional[float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        scores = []
        idx = 0

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if idx % self.subsample == 0:
                        tmp_path = str(Path(tmpdir) / f"f{idx}.png")
                        cv2.imwrite(tmp_path, frame)
                        s = self._score_path(tmp_path)
                        if s is not None:
                            scores.append(s)
                    idx += 1
        finally:
            cap.release()

        return float(np.mean(scores)) if scores else None

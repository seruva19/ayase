"""MUSIQ (Multi-Scale Image Quality Transformer) module.

No-reference IQA that handles arbitrary resolutions via a multi-
scale transformer.  Unlike CNN-based metrics, it doesn't require
fixed input sizes, making it ideal for diverse datasets.

musiq_score — higher = better quality (score range varies by model)

Uses ``pyiqa`` for pretrained MUSIQ weights.
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


class MUSIQModule(PipelineModule):
    name = "musiq"
    description = "Multi-Scale Image Quality Transformer (no-reference)"
    default_config = {
        "variant": "musiq",  # or "musiq-koniq" / "musiq-spaq"
        "subsample": 5,
        "warning_threshold": 40.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.variant = self.config.get("variant", "musiq")
        self.subsample = self.config.get("subsample", 5)
        self.warning_threshold = self.config.get("warning_threshold", 40.0)
        self._metric = None
        self._ml_available = False

    def setup(self) -> None:
        try:
            import pyiqa
            self._metric = pyiqa.create_metric(self.variant, device="cpu")
            self._ml_available = True
            logger.info(f"MUSIQ ({self.variant}) initialised")
        except ImportError:
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            logger.warning(f"Failed to setup MUSIQ: {e}")

    def _score_path(self, path: str) -> Optional[float]:
        try:
            return float(self._metric(path).item())
        except Exception as e:
            logger.debug(f"MUSIQ scoring failed: {e}")
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

            sample.quality_metrics.musiq_score = score

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low MUSIQ score: {score:.2f}",
                        details={"musiq_score": score},
                        recommendation="Multi-scale quality assessment is low.",
                    )
                )

            logger.debug(f"MUSIQ for {sample.path.name}: {score:.2f}")

        except Exception as e:
            logger.error(f"MUSIQ failed for {sample.path}: {e}")

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

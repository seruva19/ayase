"""IL-NIQE (Integrated Local NIQE) module.

IL-NIQE is an improved variant of NIQE that uses local features for
more fine-grained no-reference image quality assessment.

Range: lower = better quality (similar scale to NIQE).

Uses the ``pyiqa`` package.
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


class ILNIQEModule(PipelineModule):
    name = "ilniqe"
    description = "IL-NIQE integrated local no-reference quality (lower=better)"
    default_config = {
        "subsample": 3,
        "warning_threshold": 50.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 3)
        self.warning_threshold = self.config.get("warning_threshold", 50.0)
        self._ml_available = False
        self._metric = None

    def setup(self) -> None:
        try:
            import pyiqa

            self._metric = pyiqa.create_metric("ilniqe", device="cpu")
            self._ml_available = True
            logger.info("IL-NIQE module initialised")
        except ImportError:
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            logger.warning(f"Failed to setup IL-NIQE: {e}")

    def _score_image_path(self, path: str) -> Optional[float]:
        try:
            return float(self._metric(path).item())
        except Exception as e:
            logger.debug(f"IL-NIQE scoring failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        try:
            if sample.is_video:
                score = self._process_video(sample.path)
            else:
                score = self._score_image_path(str(sample.path))

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.ilniqe = score

            if score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High IL-NIQE (low quality): {score:.1f}",
                        details={"ilniqe": score, "threshold": self.warning_threshold},
                        recommendation="IL-NIQE indicates quality degradation in local regions.",
                    )
                )
            logger.debug(f"IL-NIQE for {sample.path.name}: {score:.1f}")
        except Exception as e:
            logger.error(f"IL-NIQE failed for {sample.path}: {e}")
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
                        s = self._score_image_path(tmp_path)
                        if s is not None:
                            scores.append(s)
                    idx += 1
        finally:
            cap.release()
        return float(np.mean(scores)) if scores else None

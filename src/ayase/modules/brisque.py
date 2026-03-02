"""BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) module.

BRISQUE is a no-reference image quality metric based on natural scene
statistics of locally normalised luminance coefficients.  It is one
of the most widely used NR-IQA metrics in the literature.

Range: roughly 0-100 (lower = better quality).
Values above ~50 typically indicate poor quality.

Uses the ``pyiqa`` package (already an Ayase dependency).
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


class BRISQUEModule(PipelineModule):
    name = "brisque"
    description = "BRISQUE no-reference image quality (lower=better)"
    default_config = {
        "subsample": 3,  # Every Nth video frame
        "warning_threshold": 50.0,  # Warn if BRISQUE > 50
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

            self._metric = pyiqa.create_metric("brisque", device="cpu")
            self._ml_available = True
            logger.info("BRISQUE module initialised")

        except ImportError:
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            logger.warning(f"Failed to setup BRISQUE: {e}")

    def _score_image_path(self, path: str) -> Optional[float]:
        """Score an image file on disk."""
        try:
            return float(self._metric(path).item())
        except Exception as e:
            logger.debug(f"BRISQUE scoring failed: {e}")
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

            sample.quality_metrics.brisque = score

            if score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High BRISQUE (low quality): {score:.1f}",
                        details={"brisque": score, "threshold": self.warning_threshold},
                        recommendation=(
                            "BRISQUE indicates significant quality degradation. "
                            "Check for noise, blur, or compression artefacts."
                        ),
                    )
                )

            logger.debug(f"BRISQUE for {sample.path.name}: {score:.1f}")

        except Exception as e:
            logger.error(f"BRISQUE failed for {sample.path}: {e}")

        return sample

    def _process_video(self, video_path: Path) -> Optional[float]:
        """Average BRISQUE across sampled video frames."""
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

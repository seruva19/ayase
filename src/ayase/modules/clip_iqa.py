"""CLIP-IQA (CLIP-based Image Quality Assessment) module.

CLIP-IQA exploits CLIP's visual-language understanding to assess
image quality without a reference image.  Unlike the existing
``clip_score`` field (which measures text–image alignment), this
metric evaluates visual quality itself using quality-related prompts.

Score range: 0-1 (higher = better perceived quality).
Uses ``pyiqa`` which ships a trained CLIP-IQA model.
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


class CLIPIQAModule(PipelineModule):
    name = "clip_iqa"
    description = "CLIP-based no-reference image quality assessment"
    default_config = {
        "subsample": 5,
        "warning_threshold": 0.4,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self.warning_threshold = self.config.get("warning_threshold", 0.4)
        self._ml_available = False
        self._metric = None

    def setup(self) -> None:
        try:
            import pyiqa

            self._metric = pyiqa.create_metric("clipiqa+", device="cpu")
            self._ml_available = True
            logger.info("CLIP-IQA+ initialised")

        except ImportError:
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            logger.warning(f"Failed to setup CLIP-IQA: {e}")

    def _score_path(self, path: str) -> Optional[float]:
        try:
            return float(self._metric(path).item())
        except Exception as e:
            logger.debug(f"CLIP-IQA scoring failed: {e}")
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

            sample.quality_metrics.clip_iqa_score = score

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low CLIP-IQA: {score:.3f}",
                        details={"clip_iqa_score": score},
                        recommendation="CLIP-based semantic quality assessment is low.",
                    )
                )

            logger.debug(f"CLIP-IQA for {sample.path.name}: {score:.3f}")

        except Exception as e:
            logger.error(f"CLIP-IQA failed for {sample.path}: {e}")

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

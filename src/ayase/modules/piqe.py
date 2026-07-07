"""PIQE (Perception-based Image Quality Evaluator) module.

PIQE is a no-reference image quality metric originally from MATLAB that
uses block-level distortion analysis. It does not require training data
and works directly on natural scene statistics.

Range: 0-100 (lower = better quality).
  0-20: Excellent, 21-35: Good, 36-50: Fair, 51-80: Poor, 81-100: Bad

Uses the ``pyiqa`` package.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PIQEModule(PipelineModule):
    name = "piqe"
    description = "PIQE perception-based no-reference quality (lower=better)"
    default_config = {
        "subsample": 3,
        "warning_threshold": 50.0,
    }
    metric_groups = {
        "piqe": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 3)
        self.warning_threshold = self.config.get("warning_threshold", 50.0)
        self._ml_available = False
        self._metric = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import pyiqa

            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._metric = pyiqa.create_metric("piqe", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("PIQE module initialised on %s", self._device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup PIQE: {e}")

    def _score_image_path(self, path: str) -> Optional[float]:
        try:
            import torch

            with torch.no_grad():
                return float(self._metric(path).item())
        except Exception as e:
            logger.debug(f"PIQE scoring failed: {e}")
            return None

    def _score_frame(self, frame_bgr: np.ndarray) -> Optional[float]:
        """Score a decoded BGR video frame via a direct RGB tensor call."""
        try:
            import torch

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(np.ascontiguousarray(rgb))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255.0
            ).to(self._device)
            with torch.no_grad():
                return float(self._metric(tensor).item())
        except Exception as e:
            logger.debug(f"PIQE frame scoring failed: {e}")
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

            sample.quality_metrics.piqe = score

            if score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High PIQE (low quality): {score:.1f}",
                        details={"piqe": score, "threshold": self.warning_threshold},
                        recommendation="PIQE indicates block-level quality issues.",
                    )
                )
            logger.debug(f"PIQE for {sample.path.name}: {score:.1f}")
        except Exception as e:
            logger.error(f"PIQE failed for {sample.path}: {e}")
        return sample

    def _process_video(self, video_path: Path) -> Optional[float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        scores = []
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % self.subsample == 0:
                    s = self._score_frame(frame)
                    if s is not None:
                        scores.append(s)
                idx += 1
        finally:
            cap.release()
        return float(np.mean(scores)) if scores else None

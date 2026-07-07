"""BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) module.

BRISQUE is a no-reference image quality metric based on natural scene
statistics of locally normalised luminance coefficients.  It is one
of the most widely used NR-IQA metrics in the literature.

Range: roughly 0-100 (lower = better quality).
Values above ~50 typically indicate poor quality.

Uses the ``pyiqa`` package (already an Ayase dependency).
"""

import logging
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
    metric_groups = {
        "brisque": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 3)
        self.warning_threshold = self.config.get("warning_threshold", 50.0)
        self._ml_available = False
        self._metric = None
        self._device = "cpu"
        self._backend = None

    def setup(self) -> None:
        try:
            import pyiqa

            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._metric = pyiqa.create_metric("brisque", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("BRISQUE module initialised on %s", self._device)

        except ImportError:
            self._backend = "unavailable"
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup BRISQUE: {e}")

    def _score_image_path(self, path: str) -> Optional[float]:
        """Score an image file on disk (pyiqa reads the real file directly)."""
        try:
            import torch

            with torch.no_grad():
                return float(self._metric(path).item())
        except Exception as e:
            logger.debug(f"BRISQUE scoring failed: {e}")
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
            )
            tensor = tensor.to(self._device)
            with torch.no_grad():
                return float(self._metric(tensor).item())
        except Exception as e:
            logger.debug(f"BRISQUE frame scoring failed: {e}")
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
        """Average BRISQUE across sampled video frames using direct tensors."""
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

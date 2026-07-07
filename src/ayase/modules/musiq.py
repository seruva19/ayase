"""MUSIQ (Multi-Scale Image Quality Transformer) module.

No-reference IQA that handles arbitrary resolutions via a multi-
scale transformer.  Unlike CNN-based metrics, it doesn't require
fixed input sizes, making it ideal for diverse datasets.

musiq_score — higher = better quality (score range varies by model)

Uses ``pyiqa`` for pretrained MUSIQ weights.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule
from ayase.runtime import resolve_torch_device

logger = logging.getLogger(__name__)


class MUSIQModule(PipelineModule):
    name = "musiq"
    description = "Multi-Scale Image Quality Transformer (no-reference)"
    default_config = {
        "variant": "musiq",  # or "musiq-koniq" / "musiq-spaq"
        "subsample": 5,
        "warning_threshold": 40.0,
    }
    metric_groups = {
        "musiq_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.variant = self.config.get("variant", "musiq")
        self.subsample = self.config.get("subsample", 5)
        self.warning_threshold = self.config.get("warning_threshold", 40.0)
        self._metric = None
        self._ml_available = False
        self._device = "cpu"
        self._backend = None

    def setup(self) -> None:
        try:
            import pyiqa

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._metric = pyiqa.create_metric(self.variant, device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("MUSIQ (%s) initialised on %s", self.variant, self._device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup MUSIQ: {e}")

    def _score_path(self, path: str) -> Optional[float]:
        try:
            import torch

            with torch.no_grad():
                return float(self._metric(path).item())
        except Exception as e:
            logger.debug(f"MUSIQ scoring failed: {e}")
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
            logger.debug(f"MUSIQ frame scoring failed: {e}")
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
        """Average MUSIQ across sampled video frames using direct tensors."""
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

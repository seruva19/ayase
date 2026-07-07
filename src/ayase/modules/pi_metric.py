"""Perceptual Index (PI) module.

The Perceptual Index was introduced as the evaluation metric for the
PIRM Challenge on Perceptual Image Super-Resolution (ECCV 2018).

PI = (10 - NRQM + NIQE) / 2

Range: lower = better perceptual quality (typically 2-8).

Uses the ``pyiqa`` package which computes PI directly.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PIModule(PipelineModule):
    name = "pi"
    description = "Perceptual Index (PIRM challenge metric, lower=better)"
    default_config = {
        "subsample": 3,
    }
    metric_groups = {
        "pi_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 3)
        self._ml_available = False
        self._metric = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import pyiqa

            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._metric = pyiqa.create_metric("pi", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("Perceptual Index (PI) module initialised on %s", self._device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup PI: {e}")

    def _score_image_path(self, path: str) -> Optional[float]:
        try:
            import torch

            with torch.no_grad():
                return float(self._metric(path).item())
        except Exception as e:
            logger.debug(f"PI scoring failed: {e}")
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
            logger.debug(f"PI frame scoring failed: {e}")
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

            sample.quality_metrics.pi_score = score
            logger.debug(f"PI for {sample.path.name}: {score:.2f}")
        except Exception as e:
            logger.error(f"PI failed for {sample.path}: {e}")
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


class PICompatModule(PIModule):
    """Compatibility alias matching filename-based discovery."""

    name = "pi_metric"

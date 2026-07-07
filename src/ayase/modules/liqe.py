"""LIQE (Learnt Image Quality Evaluator) module.

LIQE is a lightweight, fast no-reference IQA metric that jointly
predicts quality score, scene category, and distortion type in a
single forward pass. Good accuracy-to-speed trade-off.

Score range: 1-5 (higher = better, MOS-aligned).
Uses the ``pyiqa`` package.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class LIQEModule(PipelineModule):
    name = "liqe"
    description = "LIQE lightweight no-reference IQA"
    default_config = {
        "subsample": 5,
        "warning_threshold": 2.5,  # MOS-scale: 1-5
    }
    metric_groups = {
        "liqe_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self.warning_threshold = self.config.get("warning_threshold", 2.5)
        self._ml_available = False
        self._metric = None
        self._device = "cpu"
        self._backend = None

    def setup(self) -> None:
        try:
            import pyiqa
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._metric = pyiqa.create_metric("liqe", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("LIQE initialised on %s", self._device)

        except ImportError:
            self._backend = "unavailable"
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup LIQE: {e}")

    def _score_path(self, path: str) -> Optional[float]:
        try:
            return float(self._metric(path).item())
        except Exception as e:
            logger.debug(f"LIQE scoring failed: {e}")
            return None

    def _score_frame(self, frame_rgb: np.ndarray) -> Optional[float]:
        try:
            import torch

            # ascontiguousarray with a dtype change forces a fresh writable copy,
            # so the shared read-only frame-cache buffer is never touched.
            arr = np.ascontiguousarray(frame_rgb, dtype=np.float32)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 255.0
            tensor = tensor.to(self._device)
            with torch.no_grad():
                return float(self._metric(tensor).item())
        except Exception as e:
            logger.debug(f"LIQE frame scoring failed: {e}")
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

            sample.quality_metrics.liqe_score = score

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low LIQE: {score:.2f}",
                        details={"liqe_score": score},
                        recommendation="Image quality is below acceptable threshold.",
                    )
                )

            logger.debug(f"LIQE for {sample.path.name}: {score:.2f}")

        except Exception as e:
            logger.error(f"LIQE failed for {sample.path}: {e}")

        return sample

    def _process_video(self, video_path: Path) -> Optional[float]:
        frames = sample_frames(video_path, max_frames=self.subsample, color="rgb")
        scores = []
        for frame in frames:
            s = self._score_frame(frame)
            if s is not None:
                scores.append(s)
        return float(np.mean(scores)) if scores else None

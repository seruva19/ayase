"""IL-NIQE (Integrated Local NIQE) module.

IL-NIQE is an improved variant of NIQE that uses local features for
more fine-grained no-reference image quality assessment.

Range: lower = better quality (similar scale to NIQE).

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


class ILNIQEModule(PipelineModule):
    name = "ilniqe"
    description = "IL-NIQE integrated local no-reference quality (lower=better)"
    default_config = {
        "subsample": 3,
        "warning_threshold": 50.0,
    }
    metric_groups = {
        "ilniqe": "nr_quality",
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
            self._metric = pyiqa.create_metric("ilniqe", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("IL-NIQE module initialised on %s", self._device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup IL-NIQE: {e}")

    def _score_image_path(self, path: str) -> Optional[float]:
        try:
            return float(self._metric(path).item())
        except Exception as e:
            logger.debug(f"IL-NIQE scoring failed: {e}")
            return None

    def _score_frame(self, frame_rgb: np.ndarray) -> Optional[float]:
        try:
            import torch

            arr = np.ascontiguousarray(frame_rgb, dtype=np.float32)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 255.0
            tensor = tensor.to(self._device)
            with torch.no_grad():
                return float(self._metric(tensor).item())
        except Exception as e:
            logger.debug(f"IL-NIQE frame scoring failed: {e}")
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
        frames = sample_frames(video_path, max_frames=self.subsample, color="rgb")
        scores = []
        for frame in frames:
            s = self._score_frame(frame)
            if s is not None:
                scores.append(s)
        return float(np.mean(scores)) if scores else None

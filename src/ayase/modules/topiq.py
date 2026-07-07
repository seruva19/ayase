"""TOPIQ (Top-down Image Quality) module.

TOPIQ is a transformer-based no-reference image/video quality
assessment metric with strong cross-dataset generalisation.

There are two variants:
  - topiq_nr:  no-reference (standalone quality)
  - topiq_fr:  full-reference (needs reference image)

This module uses topiq_nr via the ``pyiqa`` package.
Score range: 0-1 (higher = better quality).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TOPIQModule(PipelineModule):
    name = "topiq"
    description = "TOPIQ transformer-based no-reference IQA"
    default_config = {
        "variant": "topiq_nr",  # only topiq_nr supported (NR module)
        "subsample": 5,  # Every Nth video frame
        "warning_threshold": 0.4,
    }
    metric_groups = {
        "topiq_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.variant = self.config.get("variant", "topiq_nr")
        self.subsample = self.config.get("subsample", 5)
        self.warning_threshold = self.config.get("warning_threshold", 0.4)
        self._ml_available = False
        self._metric = None
        self._backend = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.variant not in ("topiq_nr",):
            logger.warning(
                f"TOPIQ variant '{self.variant}' is not supported by this NR-only module; "
                f"falling back to 'topiq_nr'."
            )
            self.variant = "topiq_nr"

        try:
            import pyiqa
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._metric = pyiqa.create_metric(self.variant, device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info(f"TOPIQ ({self.variant}) initialised on {self._device}")

        except ImportError:
            self._backend = "unavailable"
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup TOPIQ: {e}")

    def _score_path(self, path: str) -> Optional[float]:
        try:
            return float(self._metric(path).item())
        except Exception as e:
            logger.debug(f"TOPIQ scoring failed: {e}")
            return None

    def _score_frame(self, frame_rgb: np.ndarray) -> Optional[float]:
        """Score a single RGB frame via a direct [0,1] BCHW tensor (no temp PNG)."""
        import torch

        try:
            contiguous = np.ascontiguousarray(frame_rgb)
            tensor = (
                torch.from_numpy(contiguous).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            ).to(self._device)
            return float(self._metric(tensor).item())
        except Exception as e:
            logger.debug(f"TOPIQ frame scoring failed: {e}")
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

            sample.quality_metrics.topiq_score = score

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low TOPIQ: {score:.3f}",
                        details={"topiq_score": score},
                        recommendation="Perceptual quality is low.",
                    )
                )

            logger.debug(f"TOPIQ for {sample.path.name}: {score:.3f}")

        except Exception as e:
            logger.error(f"TOPIQ failed for {sample.path}: {e}")

        return sample

    def _process_video(self, video_path: Path) -> Optional[float]:
        num_frames = int(self.config.get("num_frames", 8))
        frames = sample_frames(video_path, max_frames=max(1, num_frames), color="rgb")
        if not frames:
            return None

        scores = []
        for frame in frames:
            s = self._score_frame(frame)
            if s is not None:
                scores.append(s)

        return float(np.mean(scores)) if scores else None

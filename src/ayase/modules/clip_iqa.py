"""CLIP-IQA (CLIP-based Image Quality Assessment) module.

CLIP-IQA exploits CLIP's visual-language understanding to assess
image quality without a reference image.  Unlike the existing
``clip_score`` field (which measures text–image alignment), this
metric evaluates visual quality itself using quality-related prompts.

Score range: 0-1 (higher = better perceived quality).
Uses ``pyiqa`` which ships a trained CLIP-IQA+ model. When ``pyiqa`` is
not installed the metric is left unset (no heuristic fallback).
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ayase.image import load_representative_frame, sample_frames
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
    metric_groups = {
        "clip_iqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self.warning_threshold = self.config.get("warning_threshold", 0.4)
        self._ml_available = False
        self._metric = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import pyiqa
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._metric = pyiqa.create_metric("clipiqa+", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("CLIP-IQA+ initialised on %s", self._device)

        except ImportError:
            logger.warning("pyiqa not installed; CLIP-IQA metric skipped. Install with: pip install pyiqa")
        except Exception as e:
            logger.warning(f"Failed to setup CLIP-IQA: {e}")

    def _score_frames(self, frames: List[np.ndarray]) -> Optional[float]:
        """Score RGB frames directly as a tensor batch (no temp files)."""
        import torch

        tensors = []
        for frame in frames:
            if frame is None:
                continue
            arr = np.ascontiguousarray(frame)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            t = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)
            tensors.append(t)
        if not tensors:
            return None
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            scores = self._metric(batch)
        return float(scores.mean().item())

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            if sample.is_video:
                frames = sample_frames(sample.path, max_frames=self.subsample, color="rgb")
            else:
                frame = load_representative_frame(sample.path, color="rgb")
                frames = [frame] if frame is not None else []

            score = self._score_frames(frames)
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

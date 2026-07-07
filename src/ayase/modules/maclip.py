"""MACLIP (Multi-Attribute CLIP) module.

MACLIP is a CLIP-based no-reference image quality metric that evaluates
multiple quality attributes simultaneously using CLIP features.

Range: higher = better quality.

Uses the ``pyiqa`` package.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MACLIPModule(PipelineModule):
    name = "maclip"
    description = "MACLIP multi-attribute CLIP no-reference quality (higher=better)"
    default_config = {
        "subsample": 3,
    }
    metric_groups = {
        "maclip_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 3)
        self._ml_available = False
        self._metric = None
        self._device = "cpu"
        self._backend = None

    def setup(self) -> None:
        try:
            import pyiqa
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._metric = pyiqa.create_metric("maclip", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("MACLIP module initialised on %s", self._device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup MACLIP: {e}")

    def _score_image_path(self, path: str) -> Optional[float]:
        try:
            return float(self._metric(path).item())
        except Exception as e:
            logger.debug(f"MACLIP scoring failed: {e}")
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
            logger.debug(f"MACLIP frame scoring failed: {e}")
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

            sample.quality_metrics.maclip_score = score
            logger.debug(f"MACLIP for {sample.path.name}: {score:.3f}")
        except Exception as e:
            logger.error(f"MACLIP failed for {sample.path}: {e}")
        return sample

    def _process_video(self, video_path: Path) -> Optional[float]:
        frames = sample_frames(video_path, max_frames=self.subsample, color="rgb")
        scores = []
        for frame in frames:
            s = self._score_frame(frame)
            if s is not None:
                scores.append(s)
        return float(np.mean(scores)) if scores else None

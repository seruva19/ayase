"""MAD (Most Apparent Distortion) module."""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MADModule(PipelineModule):
    name = "mad"
    description = "Most Apparent Distortion full-reference metric (lower=better)"
    default_config = {"subsample": 8}
    metric_groups = {
        "mad": "fr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._device = "cpu"
        self._backend = None

    def setup(self) -> None:
        try:
            import pyiqa
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("mad", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("MAD model loaded on %s", self._device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("MAD unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        reference_path = getattr(sample, "reference_path", None)
        if reference_path is None:
            return sample
        try:
            import cv2
            import torch

            ref_frames = sample_frames(
                str(reference_path), max_frames=self.config.get("subsample", 8), color="rgb"
            )
            dist_frames = sample_frames(
                str(sample.path), max_frames=self.config.get("subsample", 8), color="rgb"
            )
            if not ref_frames or not dist_frames:
                return sample

            n = min(len(ref_frames), len(dist_frames))
            scores = []
            for i in range(n):
                ref_rgb = ref_frames[i]
                dist_rgb = dist_frames[i]
                h = min(ref_rgb.shape[0], dist_rgb.shape[0])
                w = min(ref_rgb.shape[1], dist_rgb.shape[1])
                # cv2.resize returns fresh writable arrays (safe vs read-only cache views)
                ref_rgb = cv2.resize(np.ascontiguousarray(ref_rgb), (w, h))
                dist_rgb = cv2.resize(np.ascontiguousarray(dist_rgb), (w, h))
                ref_t = (
                    torch.from_numpy(np.ascontiguousarray(ref_rgb, dtype=np.float32))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0
                )
                dist_t = (
                    torch.from_numpy(np.ascontiguousarray(dist_rgb, dtype=np.float32))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0
                )
                with torch.no_grad():
                    score = self._model(dist_t.to(self._device), ref_t.to(self._device)).item()
                scores.append(score)

            if scores:
                sample.quality_metrics.mad = float(np.mean(scores))
        except Exception as e:
            logger.warning("MAD processing failed: %s", e)
        return sample


class MADCompatModule(MADModule):
    """Compatibility alias matching filename-based discovery."""

    name = "mad_metric"

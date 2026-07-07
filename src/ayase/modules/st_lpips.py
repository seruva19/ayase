"""ST-LPIPS module.

Uses the real Shift-Tolerant LPIPS model from the ``stlpips-pytorch`` package
as a learned perceptual distance. Per-video the module measures a spatial
sharpness component (perceptual distance between each frame and its blurred
version) and a temporal consistency component (perceptual distance between
consecutive sampled frames), and combines them.

A previous revision silently fell back to plain LPIPS-Alex and wrote it into
the same ``st_lpips`` field. That proxy has been removed: if
``stlpips-pytorch`` is not installed the metric is reported as unavailable.
"""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class STLPIPSModule(PipelineModule):
    name = "st_lpips"
    description = "Spatiotemporal perceptual video quality (Shift-Tolerant LPIPS)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "st_lpips": "fr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._stlpips_model = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch  # noqa: F401
            import stlpips_pytorch
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._stlpips_model = stlpips_pytorch.LPIPS(net="alex").to(self._device)
            self._stlpips_model.eval()
            self._backend = "stlpips"
            self._ml_available = True
            logger.info("ST-LPIPS loaded stlpips-pytorch model on %s", self._device)
        except ImportError:
            self._backend = "unavailable"
            logger.info(
                "ST-LPIPS: 'stlpips-pytorch' not installed; metric unavailable. "
                "Install with: pip install stlpips-pytorch"
            )
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("ST-LPIPS setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            subsample = self.config.get("subsample", 8)
            frames = sample_frames(sample.path, max_frames=subsample, color="rgb")
            if len(frames) < 2:
                return sample

            spatial_scores = self._spatial_quality(frames)
            temporal_scores = self._temporal_quality(frames)

            spatial_quality = float(np.mean(spatial_scores))
            temporal_quality = float(np.mean(temporal_scores))

            # Temporal consistency weighted more heavily for video.
            st_distance = 0.4 * (1.0 - spatial_quality) + 0.6 * (1.0 - temporal_quality)
            sample.quality_metrics.st_lpips = float(np.clip(st_distance, 0.0, 1.0))
        except Exception as e:
            logger.warning("ST-LPIPS failed: %s", e)
        return sample

    def _to_tensor(self, rgb: np.ndarray):
        import cv2
        import torch

        resized = cv2.resize(np.ascontiguousarray(rgb), (256, 256))
        t = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        return t.to(self._device)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            return float(self._stlpips_model(self._to_tensor(a), self._to_tensor(b)).item())

    def _spatial_quality(self, frames) -> list:
        """Perceptual distance between each frame and its blurred version."""
        import cv2

        scores = []
        for frame in frames:
            blurred = cv2.GaussianBlur(np.ascontiguousarray(frame), (7, 7), 2.0)
            dist = self._distance(frame, blurred)
            # Higher distance = more detail lost by blurring = sharper original.
            scores.append(float(min(1.0, dist * 5.0)))
        return scores

    def _temporal_quality(self, frames) -> list:
        """Perceptual consistency between consecutive sampled frames."""
        scores = []
        for i in range(len(frames) - 1):
            dist = self._distance(frames[i], frames[i + 1])
            scores.append(float(1.0 / (1.0 + dist * 3.0)))
        return scores

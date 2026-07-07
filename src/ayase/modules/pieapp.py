"""PieAPP (Perceptual Image-Error Assessment through Pairwise Preference) module."""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PieAPPModule(PipelineModule):
    name = "pieapp"
    description = "PieAPP full-reference perceptual error via pairwise preference (lower=better)"
    default_config = {"subsample": 8}
    metric_groups = {
        "pieapp": "fr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import pyiqa
            import torch

            from ayase.runtime import resolve_torch_device

            device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("pieapp", device=device)
            try:
                self._device = next(self._model.parameters()).device
            except StopIteration:
                self._device = torch.device(device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("PieAPP model loaded on %s", device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("PieAPP unavailable: pyiqa not installed")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("PieAPP unavailable: %s", e)

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

            ref_frames = self._load_frames_from(str(reference_path), sample.is_video)
            dist_frames = self._load_frames_from(str(sample.path), sample.is_video)
            if not ref_frames or not dist_frames:
                return sample

            n = min(len(ref_frames), len(dist_frames))
            scores = []
            device = self._device
            for i in range(n):
                ref_rgb = cv2.cvtColor(ref_frames[i], cv2.COLOR_BGR2RGB)
                dist_rgb = cv2.cvtColor(dist_frames[i], cv2.COLOR_BGR2RGB)
                h = min(ref_rgb.shape[0], dist_rgb.shape[0])
                w = min(ref_rgb.shape[1], dist_rgb.shape[1])
                ref_rgb = cv2.resize(ref_rgb, (w, h))
                dist_rgb = cv2.resize(dist_rgb, (w, h))
                ref_t = torch.from_numpy(ref_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                dist_t = torch.from_numpy(dist_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    score = self._model(dist_t.to(device), ref_t.to(device)).item()
                scores.append(score)

            sample.quality_metrics.pieapp = float(np.mean(scores))
        except Exception as e:
            logger.warning("PieAPP processing failed: %s", e)
        return sample

    def _load_frames_from(self, path: str, is_video: bool) -> list:
        from ayase.image import sample_frames

        subsample = self.config.get("subsample", 8)
        return sample_frames(path, max_frames=subsample, color="bgr")

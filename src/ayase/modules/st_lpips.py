"""ST-LPIPS (SpatioTemporal LPIPS) module.

Computes spatiotemporal perceptual video quality using the ST-LPIPS model
from the ``stlpips-pytorch`` package when available.

Backend tiers:
  1. **stlpips-pytorch** — real ST-LPIPS model (``pip install stlpips-pytorch``)
  2. **lpips** — standard LPIPS-Alex applied to consecutive frames
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class STLPIPSModule(PipelineModule):
    name = "st_lpips"
    description = "Spatiotemporal perceptual video quality (ST-LPIPS model or LPIPS)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._stlpips_model = None
        self._lpips_model = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: Real ST-LPIPS from stlpips-pytorch
        try:
            import torch
            import stlpips_pytorch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._stlpips_model = stlpips_pytorch.LPIPS(net="alex").to(device)
            self._stlpips_model.eval()
            self._device = device
            self._backend = "stlpips"
            self._ml_available = True
            logger.info("ST-LPIPS loaded real stlpips-pytorch model on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.info("stlpips-pytorch unavailable: %s", e)

        # Tier 2: Standard LPIPS-Alex
        try:
            import torch
            import lpips

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._lpips_model = lpips.LPIPS(net="alex").to(device)
            self._lpips_model.eval()
            self._device = device
            self._backend = "lpips"
            self._ml_available = True
            logger.info("ST-LPIPS using LPIPS-Alex fallback on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.warning("ST-LPIPS: no ML backend available: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not sample.is_video:
            return sample
        try:
            import cv2

            subsample = self.config.get("subsample", 8)
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()

            if len(frames) < 2:
                return sample

            # Spatial component: per-frame quality
            spatial_scores = self._spatial_quality(frames)

            # Temporal component: inter-frame perceptual consistency
            temporal_scores = self._temporal_quality(frames)

            spatial_quality = float(np.mean(spatial_scores))
            temporal_quality = float(np.mean(temporal_scores))

            # Weighted combination (temporal matters more for video)
            st_distance = 0.4 * (1.0 - spatial_quality) + 0.6 * (1.0 - temporal_quality)

            sample.quality_metrics.st_lpips = float(np.clip(st_distance, 0.0, 1.0))
        except Exception as e:
            logger.warning("ST-LPIPS failed: %s", e)
        return sample

    def _spatial_quality(self, frames) -> list:
        """Compute per-frame spatial perceptual quality using the best backend."""
        if self._backend == "stlpips":
            return self._spatial_quality_stlpips(frames)
        return self._spatial_quality_lpips(frames)

    def _spatial_quality_lpips_impl(self, frames, model) -> list:
        """LPIPS-based spatial quality: distance between frame and its blurred version."""
        import cv2
        import torch

        target_size = (256, 256)
        scores = []
        for frame in frames:
            rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), target_size)
            blurred = cv2.GaussianBlur(rgb, (7, 7), 2.0)

            t_orig = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            t_blur = torch.from_numpy(blurred).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            t_orig, t_blur = t_orig.to(self._device), t_blur.to(self._device)

            with torch.no_grad():
                dist = model(t_orig, t_blur).item()
            # Higher dist = more detail lost by blurring = sharper original
            scores.append(float(min(1.0, dist * 5.0)))
        return scores

    def _spatial_quality_stlpips(self, frames) -> list:
        """Spatial quality using real ST-LPIPS model."""
        return self._spatial_quality_lpips_impl(frames, self._stlpips_model)

    def _spatial_quality_lpips(self, frames) -> list:
        """Spatial quality using standard LPIPS model."""
        return self._spatial_quality_lpips_impl(frames, self._lpips_model)

    def _temporal_quality(self, frames) -> list:
        """Compute temporal perceptual consistency using the best available backend."""
        if self._backend == "stlpips":
            return self._temporal_quality_stlpips(frames)
        return self._temporal_quality_lpips(frames)

    def _temporal_quality_stlpips(self, frames) -> list:
        """Compute temporal quality using real ST-LPIPS model."""
        import cv2
        import torch

        target_size = (256, 256)
        scores = []

        for i in range(len(frames) - 1):
            f1 = cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), target_size)
            f2 = cv2.resize(cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2RGB), target_size)

            t1 = torch.from_numpy(f1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            t2 = torch.from_numpy(f2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

            t1, t2 = t1.to(self._device), t2.to(self._device)

            with torch.no_grad():
                dist = self._stlpips_model(t1, t2).item()

            consistency = 1.0 / (1.0 + dist * 3.0)
            scores.append(float(consistency))

        return scores

    def _temporal_quality_lpips(self, frames) -> list:
        """Compute temporal quality using standard LPIPS on consecutive frames."""
        import cv2
        import torch

        target_size = (256, 256)
        scores = []

        for i in range(len(frames) - 1):
            f1 = cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), target_size)
            f2 = cv2.resize(cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2RGB), target_size)

            t1 = torch.from_numpy(f1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            t2 = torch.from_numpy(f2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

            t1, t2 = t1.to(self._device), t2.to(self._device)

            with torch.no_grad():
                spatial_dist = self._lpips_model(t1, t2).item()

            consistency = 1.0 / (1.0 + spatial_dist * 3.0)
            scores.append(float(consistency))

        return scores


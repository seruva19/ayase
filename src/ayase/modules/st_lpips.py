"""ST-LPIPS (SpatioTemporal LPIPS) module.

Computes spatiotemporal perceptual video quality using the ST-LPIPS model
from the ``stlpips-pytorch`` package when available.

Backend tiers:
  1. **stlpips-pytorch** — real ST-LPIPS model (``pip install stlpips-pytorch``)
  2. **lpips** — standard LPIPS-Alex applied to consecutive frames
  3. **heuristic** — OpenCV gradient/flow-based structural proxy
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class STLPIPSModule(PipelineModule):
    name = "st_lpips"
    description = "Spatiotemporal perceptual video quality (ST-LPIPS model, LPIPS, or heuristic fallback)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = "heuristic"
        self._stlpips_model = None
        self._lpips_model = None
        self._device = None

    def setup(self) -> None:
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
            logger.info("LPIPS unavailable, using heuristic: %s", e)

        # Tier 3: heuristic (no ML needed)
        self._backend = "heuristic"
        self._ml_available = True

    def process(self, sample: Sample) -> Sample:
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
            st_quality = 0.4 * spatial_quality + 0.6 * temporal_quality

            sample.quality_metrics.st_lpips = float(np.clip(st_quality, 0.0, 1.0))
        except Exception as e:
            logger.warning("ST-LPIPS failed: %s", e)
        return sample

    def _spatial_quality(self, frames) -> list:
        """Compute per-frame spatial perceptual quality."""
        import cv2

        scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

            # Multi-scale structural features
            features = []
            current = gray.copy()
            for scale in range(3):
                lap_var = cv2.Laplacian(current, cv2.CV_64F).var()
                features.append(min(1.0, lap_var / 500.0))
                h, w = current.shape
                current = cv2.resize(current, (max(1, w // 2), max(1, h // 2)))

            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)
            complexity = min(1.0, np.mean(grad_mag) / 50.0)
            features.append(complexity)

            scores.append(float(np.mean(features)))
        return scores

    def _temporal_quality(self, frames) -> list:
        """Compute temporal perceptual consistency using the best available backend."""
        if self._backend == "stlpips":
            return self._temporal_quality_stlpips(frames)
        elif self._backend == "lpips":
            return self._temporal_quality_lpips(frames)
        return self._temporal_quality_heuristic(frames)

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

    def _temporal_quality_heuristic(self, frames) -> list:
        """Compute temporal quality using optical flow heuristic."""
        import cv2

        scores = []
        for i in range(len(frames) - 1):
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float64)
            g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(np.float64)

            flow = cv2.calcOpticalFlowFarneback(
                g1.astype(np.uint8), g2.astype(np.uint8),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            h, w = g1.shape
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (map_x + flow[..., 0]).astype(np.float32)
            map_y = (map_y + flow[..., 1]).astype(np.float32)
            warped = cv2.remap(g1.astype(np.float32), map_x, map_y,
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            residual = np.mean(np.abs(warped - g2.astype(np.float32)))
            consistency = 1.0 / (1.0 + residual / 20.0)

            dt = g2 - g1
            dt_grad = cv2.Sobel(dt.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
            smoothness = 1.0 / (1.0 + np.std(dt_grad) / 20.0)

            scores.append(0.6 * consistency + 0.4 * smoothness)

        return scores

"""Flow-compensated perceptual distance (FloLPIPS).

Computes optical flow between consecutive frames, warps one frame to align
with the next, then measures perceptual distance. Assesses temporal
consistency of visual quality.

Backend tiers:
  1. **RAFT + LPIPS** — torchvision RAFT-Small optical flow + LPIPS-Alex
  2. **Farneback + LPIPS** — OpenCV Farneback flow + LPIPS-Alex
  3. **Farneback + MSE** — OpenCV Farneback flow + MSE distance
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FloLPIPSModule(PipelineModule):
    name = "flolpips"
    description = "Flow-compensated perceptual distance (RAFT+LPIPS, Farneback+LPIPS, or MSE fallback)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = "farneback_mse"
        self._lpips_model = None
        self._raft_model = None
        self._raft_transforms = None
        self._device = None

    def setup(self) -> None:
        import_ok = False
        device = None

        # Check torch availability first
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            import_ok = True
        except ImportError:
            pass

        # Try loading RAFT
        raft_ok = False
        if import_ok:
            try:
                from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
                weights = Raft_Small_Weights.DEFAULT
                self._raft_model = raft_small(weights=weights).to(device).eval()
                self._raft_transforms = weights.transforms()
                raft_ok = True
                logger.info("FloLPIPS loaded RAFT-Small optical flow on %s", device)
            except (ImportError, Exception) as e:
                logger.info("RAFT unavailable, using Farneback: %s", e)

        # Try loading LPIPS
        lpips_ok = False
        if import_ok:
            try:
                import lpips
                self._lpips_model = lpips.LPIPS(net="alex").to(device)
                self._lpips_model.eval()
                lpips_ok = True
            except (ImportError, Exception) as e:
                logger.info("LPIPS unavailable, using MSE: %s", e)

        self._device = device

        if raft_ok and lpips_ok:
            self._backend = "raft_lpips"
        elif lpips_ok:
            self._backend = "farneback_lpips"
        else:
            self._backend = "farneback_mse"

        self._ml_available = True
        logger.info("FloLPIPS backend: %s", self._backend)

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

            flow_lpips_scores = []
            for i in range(len(frames) - 1):
                score = self._compute_flow_lpips(frames[i], frames[i + 1])
                flow_lpips_scores.append(score)

            mean_dist = float(np.mean(flow_lpips_scores))

            sample.quality_metrics.flolpips = float(mean_dist)
        except Exception as e:
            logger.warning("FloLPIPS failed: %s", e)
        return sample

    def _compute_flow(self, frame1, frame2):
        """Compute optical flow using the best available backend."""
        import cv2

        if self._backend == "raft_lpips" and self._raft_model is not None:
            try:
                return self._compute_raft_flow(frame1, frame2)
            except Exception as e:
                logger.debug("RAFT flow failed, falling back to Farneback: %s", e)

        # Farneback fallback
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        return flow

    def _compute_raft_flow(self, frame1, frame2):
        """Compute optical flow using RAFT-Small."""
        import torch
        import cv2

        h, w = frame1.shape[:2]
        # RAFT needs images as float tensors [B, 3, H, W] in [0, 255]
        rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        t1 = torch.from_numpy(rgb1).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
        t2 = torch.from_numpy(rgb2).permute(2, 0, 1).unsqueeze(0).float().to(self._device)

        # Apply transforms
        if self._raft_transforms is not None:
            t1, t2 = self._raft_transforms(t1, t2)

        with torch.no_grad():
            # RAFT returns list of flow predictions (multi-scale), take last
            flow_predictions = self._raft_model(t1, t2)
            flow = flow_predictions[-1]  # [B, 2, H, W]

        # Convert to numpy [H, W, 2]
        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()

        # Resize to original frame size if needed
        fh, fw = flow_np.shape[:2]
        if fh != h or fw != w:
            import cv2 as cv
            flow_np = cv.resize(flow_np, (w, h))

        return flow_np

    def _compute_flow_lpips(self, frame1, frame2) -> float:
        """Compute flow-compensated perceptual distance between frames."""
        import cv2

        flow = self._compute_flow(frame1, frame2)

        # Warp frame1 using flow to predict frame2
        h, w = frame1.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + flow[..., 0]).astype(np.float32)
        map_y = (map_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

        if self._lpips_model is not None:
            return self._lpips_distance(warped, frame2)
        else:
            return self._mse_distance(warped, frame2)

    def _lpips_distance(self, img1, img2) -> float:
        """Compute LPIPS perceptual distance."""
        import torch
        import cv2

        target_size = (256, 256)
        rgb1 = cv2.cvtColor(cv2.resize(img1, target_size), cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(cv2.resize(img2, target_size), cv2.COLOR_BGR2RGB)

        t1 = torch.from_numpy(rgb1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        t2 = torch.from_numpy(rgb2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

        t1, t2 = t1.to(self._device), t2.to(self._device)

        with torch.no_grad():
            dist = self._lpips_model(t1, t2).item()

        return float(dist)

    def _mse_distance(self, img1, img2) -> float:
        """Fallback MSE-based perceptual distance."""
        diff = img1.astype(np.float64) - img2.astype(np.float64)
        mse = np.mean(diff ** 2) / (255.0 ** 2)
        return float(mse)

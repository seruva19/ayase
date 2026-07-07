"""FloLPIPS — flow-weighted LPIPS (Danier et al. 2022).

Full-reference video quality metric: for each aligned (reference, distorted)
frame pair it computes the LPIPS *spatial* error map and averages it weighted by
the optical-flow magnitude between consecutive reference frames, so distortions
in moving regions count more. Optical flow comes from RAFT-Small (torchvision,
shared with other RAFT users) and the perceptual map from LPIPS-Alex.

Requires a reference video plus real RAFT and LPIPS. When any of those is
missing the metric is left unset — there is no Farneback/MSE stand-in.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FloLPIPSModule(PipelineModule):
    name = "flolpips"
    description = "Flow-weighted LPIPS full-reference video quality (RAFT + LPIPS)"
    default_config = {
        "subsample": 8,
        "size": 256,
    }
    metric_groups = {
        "flolpips": "fr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend = "unavailable"
        self._lpips_model = None
        self._raft_model = None
        self._raft_transforms = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import os
            import torch  # noqa: F401
            from ayase.runtime import resolve_torch_device, shared_runtime_resource

            models_dir = self.config.get("models_dir")
            if models_dir:
                os.environ.setdefault("TORCH_HOME", str(models_dir))

            self._device = resolve_torch_device(self.config.get("device", "auto"))

            # Real LPIPS (spatial map) — required.
            import lpips

            def load_lpips():
                return lpips.LPIPS(net="alex", spatial=True).to(self._device).eval()

            self._lpips_model = shared_runtime_resource(
                self,
                ("lpips_alex_spatial", str(self._device)),
                load_lpips,
            )

            # Real RAFT-Small optical flow — shared with other RAFT users.
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

            def load_raft():
                weights = Raft_Small_Weights.DEFAULT
                model = raft_small(weights=weights, progress=False).to(self._device)
                model.eval()
                return model, weights.transforms()

            self._raft_model, self._raft_transforms = shared_runtime_resource(
                self,
                ("raft", "raft_small", str(self._device)),
                load_raft,
            )

            self._backend = "raft_lpips"
            logger.info("FloLPIPS initialised (RAFT-Small + LPIPS-Alex) on %s", self._device)
        except ImportError as e:
            logger.warning("FloLPIPS unavailable: requires torchvision RAFT and the 'lpips' package (%s)", e)
        except Exception as e:
            logger.warning("FloLPIPS unavailable: backend load failed (%s)", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if self._backend != "raft_lpips" or not sample.is_video:
            return sample

        # FloLPIPS is full-reference; without a reference video there is no metric.
        reference_path = getattr(sample, "reference_path", None)
        if reference_path is None or not Path(str(reference_path)).exists():
            return sample

        try:
            import torch

            size = int(self.config.get("size", 256))
            ref_frames = self._load_consecutive(Path(str(reference_path)), size)
            dist_frames = self._load_consecutive(sample.path, size)
            n = min(len(ref_frames), len(dist_frames))
            if n < 2:
                return sample

            scores = []
            with torch.no_grad():
                for i in range(n - 1):
                    flow_mag = self._flow_magnitude(ref_frames[i], ref_frames[i + 1])
                    error_map = self._lpips_spatial(ref_frames[i], dist_frames[i])
                    if flow_mag.shape != error_map.shape:
                        import cv2

                        flow_mag = cv2.resize(
                            flow_mag, (error_map.shape[1], error_map.shape[0])
                        )
                    weight_sum = float(flow_mag.sum())
                    if weight_sum > 1e-8:
                        scores.append(float((flow_mag * error_map).sum() / weight_sum))
                    else:
                        scores.append(float(error_map.mean()))

            if scores:
                sample.quality_metrics.flolpips = float(np.mean(scores))
        except Exception as e:
            logger.warning("FloLPIPS failed: %s", e)
        return sample

    def _flow_magnitude(self, rgb1: np.ndarray, rgb2: np.ndarray) -> np.ndarray:
        """RAFT optical-flow magnitude map between two reference frames."""
        import torch

        t1 = torch.from_numpy(np.ascontiguousarray(rgb1)).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
        t2 = torch.from_numpy(np.ascontiguousarray(rgb2)).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
        if self._raft_transforms is not None:
            t1, t2 = self._raft_transforms(t1, t2)
        flow = self._raft_model(t1, t2)[-1]  # [1, 2, H, W]
        mag = torch.norm(flow.squeeze(0), dim=0)  # [H, W]
        return mag.detach().cpu().numpy()

    def _lpips_spatial(self, rgb1: np.ndarray, rgb2: np.ndarray) -> np.ndarray:
        """LPIPS per-pixel perceptual distance map (reference vs distorted)."""
        import torch

        t1 = torch.from_numpy(np.ascontiguousarray(rgb1)).permute(2, 0, 1).unsqueeze(0).float().to(self._device) / 127.5 - 1.0
        t2 = torch.from_numpy(np.ascontiguousarray(rgb2)).permute(2, 0, 1).unsqueeze(0).float().to(self._device) / 127.5 - 1.0
        dist_map = self._lpips_model(t1, t2)  # [1, 1, H, W] with spatial=True
        return dist_map.squeeze().detach().cpu().numpy()

    def _load_consecutive(self, path: Path, size: int) -> List[np.ndarray]:
        """Decode the first N consecutive native-fps frames as RGB, resized square."""
        import cv2

        n_want = int(self.config.get("subsample", 8)) + 1
        frames: List[np.ndarray] = []
        cap = cv2.VideoCapture(str(path))
        try:
            while len(frames) < n_want:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(rgb, (size, size)))
        finally:
            cap.release()
        return frames

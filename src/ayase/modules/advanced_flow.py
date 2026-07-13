"""RAFT optical flow magnitude analysis across all consecutive frame pairs.

Uses RAFT-Large or RAFT-Small from torchvision to compute mean flow score.
Higher flow_score indicates more motion. Low scores flag static content."""

import logging
import numpy as np
import cv2
from typing import List

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_RAFT_MIRRORS = {
    "raft_large": (
        "raft_large_C_T_SKHT_V2-ff5fadd5.pth",
        "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/"
        "advanced_flow/raft_large_C_T_SKHT_V2-ff5fadd5.pth",
    ),
    "raft_small": (
        "raft_small_C_T_V2-01064c6d.pth",
        "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/"
        "advanced_flow/raft_small_C_T_V2-01064c6d.pth",
    ),
}


def _cap_frame_resolution(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale a frame so its longer side <= max_side, keeping dims divisible by 8.

    RAFT's correlation volume grows ~O((H*W)^2); on HD frames this exhausts GPU
    memory (observed: a single 1080p pair tried to allocate 62 GiB). Only oversized
    frames are downscaled; smaller frames pass through unchanged. RAFT requires the
    spatial dims to be multiples of 8, so the resized dimensions are floored to /8.
    """
    if max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    longer = max(h, w)
    if longer <= max_side:
        return frame
    scale = max_side / longer
    new_w = max(8, (int(round(w * scale)) // 8) * 8)
    new_h = max(8, (int(round(h * scale)) // 8) * 8)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class AdvancedFlowModule(PipelineModule):
    name = "advanced_flow"
    description = "RAFT optical flow: flow_score (all consecutive pairs)"

    default_config = {
        "use_large_model": True,
        "max_frames": 150,
        "max_resolution": 512,
    }
    metric_groups = {
        "flow_score": "motion",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.use_large_model = self.config.get("use_large_model", True)
        self.max_frames = self.config.get("max_frames", 150)
        self.max_resolution = self.config.get("max_resolution", 512)
        self._model = None
        self._device = "cpu"
        self._ml_available = False
        self._transforms = None
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import os
            from ayase.runtime import resolve_torch_device, shared_runtime_resource
            from ayase.config import download_torch_hub_checkpoint

            # Redirect torch hub cache to models_dir so RAFT weights respect config
            models_dir = str(self.config.get("models_dir", "models"))
            os.environ["TORCH_HOME"] = models_dir

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            variant = "raft_large" if self.use_large_model else "raft_small"
            filename, url = _RAFT_MIRRORS[variant]
            download_torch_hub_checkpoint(filename, url, models_dir)

            def load_raft():
                if self.use_large_model:
                    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
                    weights = Raft_Large_Weights.DEFAULT
                    model = raft_large(weights=weights, progress=False).to(self._device)
                else:
                    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
                    weights = Raft_Small_Weights.DEFAULT
                    model = raft_small(weights=weights, progress=False).to(self._device)
                model.eval()
                return model, weights.transforms()

            logger.info("Setting up RAFT (%s) on %s...", variant, self._device)
            self._model, self._transforms = shared_runtime_resource(
                self,
                ("raft", variant, str(self._device)),
                load_raft,
            )
            self._backend = variant
            self._ml_available = True

        except ImportError:
            logger.warning("torchvision >= 0.13 required for RAFT.")
        except Exception as e:
            logger.warning(f"Failed to setup RAFT: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            import torch

            frames = self._load_all_frames(sample)
            if len(frames) < 2:
                return sample

            # Compute flow for ALL consecutive frame pairs (matches EvalCrafter)
            optical_flows = []

            with torch.no_grad():
                for i in range(len(frames) - 1):
                    img1 = torch.from_numpy(frames[i]).permute(2, 0, 1).unsqueeze(0).to(self._device)
                    img2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).unsqueeze(0).to(self._device)

                    if self._transforms:
                        img1, img2 = self._transforms(img1, img2)

                    list_of_flows = self._model(img1, img2)
                    predicted_flow = list_of_flows[-1]

                    flow_magnitude = torch.norm(predicted_flow.squeeze(0), dim=0)
                    mean_flow = flow_magnitude.mean().item()
                    optical_flows.append(mean_flow)

            if not optical_flows:
                return sample

            flow_score = float(np.mean(optical_flows))

            if sample.quality_metrics is None:
                from ayase.models import QualityMetrics
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.flow_score = flow_score

            if flow_score < 0.5:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Low Dynamic Degree (Static): {flow_score:.2f}",
                        details={"flow_score": flow_score},
                    )
                )

        except Exception as e:
            logger.warning(f"Flow analysis failed: {e}")

        return sample

    def _load_all_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
        cap = None
        try:
            cap = cv2.VideoCapture(str(sample.path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames > 0 and total_frames > self.max_frames:
                # Subsample uniformly to stay within max_frames
                indices = set(np.linspace(0, total_frames - 1, self.max_frames, dtype=int))
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx in indices:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(_cap_frame_resolution(frame, self.max_resolution))
                    frame_idx += 1
            else:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(_cap_frame_resolution(frame, self.max_resolution))
                    if len(frames) >= self.max_frames:
                        break
        except Exception as e:
            logger.debug(f"Failed to load frames for advanced flow: {e}")
        finally:
            if cap is not None:
                cap.release()
        return frames

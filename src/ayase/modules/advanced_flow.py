"""EvalCrafter Flow Score — mean RAFT optical-flow magnitude.

``flow_score`` is the mean dense-flow magnitude over every consecutive frame
pair, computed by RAFT (raft-things weights, iters=20, native resolution).
Higher means more pixel displacement, not better quality; values depend on
resolution and frame rate and are not directly comparable across unlike
inputs. ``max_frames``/``max_resolution`` above zero are explicit non-default
caps for memory-constrained runs.
"""

import logging
import numpy as np
import cv2
from typing import List

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_RAFT_CTV1_FILE = "raft_large_C_T_V1-22a6c225.pth"
_RAFT_CTV1_MIRROR = (
    "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/"
    f"advanced_flow/{_RAFT_CTV1_FILE}"
)


def _pad_to_multiple_of_8(img: "object") -> "object":
    """Replicate-pad a CHW float tensor so H and W are multiples of 8.

    Same role as RAFT's ``InputPadder`` — EvalCrafter pads rather than
    downscales, so flow magnitude keeps its native scale.
    """
    import torch.nn.functional as F

    h, w = img.shape[-2:]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if not pad_h and not pad_w:
        return img
    return F.pad(img, (0, pad_w, 0, pad_h), mode="replicate")


def _cap_frame_resolution(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Optionally downscale a frame so its longer side <= max_side.

    Disabled when ``max_side <= 0`` (the default — EvalCrafter evaluates at
    native resolution). An explicit positive value is a caller-side memory
    guard for very large frames where the RAFT correlation volume would OOM;
    the output is rounded to multiples of 8 like the RAFT input padder.
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
    provenance = "published"
    sources = {
        "flow_score": "EvalCrafter Flow Score (RAFT) — https://github.com/evalcrafter/EvalCrafter/blob/master/metrics/RAFT/optical_flow_scores.py",
    }
    description = "RAFT optical flow: flow_score, mean magnitude over all consecutive pairs (EvalCrafter)"

    default_config = {
        "max_frames": 0,
        "max_resolution": 0,
    }
    metric_groups = {
        "flow_score": "motion",
    }

    def __init__(self, config=None):
        super().__init__(config)
        # EvalCrafter evaluates every consecutive pair at native resolution;
        # positive values are explicit non-default memory caps.
        self.max_frames = self.config.get("max_frames", 0)
        self.max_resolution = self.config.get("max_resolution", 0)
        self._model = None
        self._device = "cpu"
        self._ml_available = False
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
            try:
                download_torch_hub_checkpoint(_RAFT_CTV1_FILE, _RAFT_CTV1_MIRROR, models_dir)
            except Exception:
                pass  # torchvision falls back to download.pytorch.org

            def load_raft():
                from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

                # C_T_V1 is torchvision's port of the official raft-things.pth
                # (Chairs -> FlyingThings3D) — the EvalCrafter checkpoint.
                weights = Raft_Large_Weights.C_T_V1
                model = raft_large(weights=weights, progress=False).to(self._device)
                model.eval()
                return model

            logger.info("Setting up RAFT (things weights) on %s...", self._device)
            self._model = shared_runtime_resource(
                self,
                ("raft", "raft_large_things", str(self._device)),
                load_raft,
            )
            self._backend = "raft_large"
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
                    img1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float().unsqueeze(0).to(self._device)
                    img2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).float().unsqueeze(0).to(self._device)

                    # EvalCrafter: InputPadder (replicate pad to x8), no resize;
                    # torchvision RAFT expects inputs normalized to [-1, 1].
                    img1 = _pad_to_multiple_of_8(img1 / 255.0 * 2.0 - 1.0)
                    img2 = _pad_to_multiple_of_8(img2 / 255.0 * 2.0 - 1.0)

                    predicted_flow = self._model(img1, img2, num_flow_updates=20)[-1]

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

            if self.max_frames > 0 and total_frames > self.max_frames:
                # Explicit non-default cap: uniform subsample.
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
                    if self.max_frames > 0 and len(frames) >= self.max_frames:
                        break
        except Exception as e:
            logger.debug(f"Failed to load frames for advanced flow: {e}")
        finally:
            if cap is not None:
                cap.release()
        return frames

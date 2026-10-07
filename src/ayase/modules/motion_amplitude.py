"""EvalCrafter Motion AC-Score (motion_ac_score).

Computes mean RAFT optical-flow magnitude over all consecutive frame pairs
(full resolution, raft-things-equivalent weights, iters=20), classifies the
video as "large" (mean > 5) or "slow", and returns the binary match against
the expected amplitude — 1.0 on match, 0.0 on mismatch (EvalCrafter scale).

Expected amplitude comes from the ``expected_motion`` config or caption
keywords (EvalCrafter takes it from benchmark metadata — documented
deviation).
"""

import logging
from typing import Optional, List

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _cap_frame_resolution(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Optionally downscale an HWC frame so the long side <= ``max_side``.

    Disabled when ``max_side <= 0`` (the default — EvalCrafter evaluates at
    native resolution). An explicit positive value is a caller-side memory
    guard for very large frames where the RAFT correlation volume would OOM;
    the output is rounded to multiples of 8 like the RAFT input padder.
    """
    if max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return frame
    scale = max_side / side
    new_w = max(8, (int(w * scale) // 8) * 8)
    new_h = max(8, (int(h * scale) // 8) * 8)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _pad_to_multiple_of_8(img: "object") -> "object":
    """Replicate-pad a CHW float tensor so H and W are multiples of 8.

    Same role as RAFT's ``InputPadder`` — EvalCrafter pads rather than
    downscales, so flow magnitude keeps its native scale for the threshold.
    """
    import torch
    import torch.nn.functional as F

    h, w = img.shape[-2:]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if not pad_h and not pad_w:
        return img
    return F.pad(img, (0, pad_w, 0, pad_h), mode="replicate")


# Keywords that indicate fast/large motion in captions
FAST_KEYWORDS = {
    "fast", "quick", "rapid", "running", "sprinting", "rushing",
    "racing", "flying", "speeding", "dashing", "jumping", "exploding",
    "crashing", "falling", "spinning", "whipping", "zooming",
    "dancing", "fighting", "chasing",
}

# Keywords that indicate slow/static motion in captions
SLOW_KEYWORDS = {
    "slow", "static", "still", "calm", "gentle", "steady",
    "standing", "sitting", "resting", "floating", "drifting",
    "walking slowly", "relaxing", "peaceful", "sleeping",
}


class MotionAmplitudeModule(PipelineModule):
    name = "motion_amplitude"
    provenance = "adapted"
    sources = {
        "motion_ac_score": "EvalCrafter Motion AC-Score (Liu et al., CVPR 2024) — https://github.com/evalcrafter/EvalCrafter/blob/master/metrics/RAFT/optical_flow_scores.py",
    }
    deviations = {
        "motion_ac_score": "Uses torchvision's port of RAFT things weights with corrected input normalization; expected motion may be inferred from caption keywords instead of EvalCrafter metadata, and max_side can explicitly downsample inputs",
    }
    description = "Motion amplitude classification vs expected label (EvalCrafter motion_ac_score via RAFT)"

    default_config = {
        "amplitude_threshold": 5.0,
        "max_frames": 0,
        "max_resolution": 0,
    }
    metric_groups = {
        "motion_ac_score": "motion",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.amplitude_threshold = self.config.get("amplitude_threshold", 5.0)
        self.max_frames = self.config.get("max_frames", 0)
        # 0 = native resolution (EvalCrafter protocol). An explicit positive
        # value caps the long side — a memory guard, not the protocol.
        self.max_resolution = self.config.get("max_resolution", 0)
        self._model = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        try:
            import os
            from ayase.runtime import resolve_torch_device, shared_runtime_resource

            # Redirect torch hub cache to models_dir so RAFT weights respect config
            models_dir = self.config.get("models_dir")
            if models_dir:
                os.environ["TORCH_HOME"] = str(models_dir)

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            logger.info(f"Setting up RAFT (things weights) for motion amplitude on {self._device}...")

            def load_raft():
                from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

                # C_T_V1 is torchvision's port of the official raft-things.pth
                # (Chairs -> FlyingThings3D) — the EvalCrafter checkpoint.
                weights = Raft_Large_Weights.C_T_V1
                model = raft_large(weights=weights, progress=False).to(self._device)
                model.eval()
                return model

            # Share the RAFT model with other RAFT-based modules.
            self._model = shared_runtime_resource(
                self, ("raft", "raft_large_things", str(self._device)), load_raft
            )
            self._ml_available = True
            self._backend = "raft_large"

        except ImportError:
            self._backend = "unavailable"
            logger.warning("torchvision >= 0.13 required for RAFT.")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup RAFT: {e}")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or not self._ml_available:
            return sample

        caption_text = None
        if sample.caption:
            caption_text = sample.caption.text
        else:
            txt_path = sample.path.with_suffix(".txt")
            if txt_path.exists():
                try:
                    caption_text = txt_path.read_text().strip()
                except Exception:
                    pass

        # Prefer explicit expected_motion from config (set by downstream caller)
        # Accepts "large"/"fast" or "slow"/"small"
        explicit = self.config.get("expected_motion")
        if explicit:
            expected = "large" if explicit.lower() in ("large", "fast", "medium") else "slow"
        else:
            if not caption_text:
                return sample
            expected = self._classify_caption_motion(caption_text)

        if expected is None:
            return sample

        try:
            mean_flow = self._compute_mean_flow(sample)
            if mean_flow is None:
                return sample

            predicted = "large" if abs(mean_flow) > self.amplitude_threshold else "slow"

            # EvalCrafter: amp_recognition_score is a binary 1/0 match.
            score = 1.0 if predicted == expected else 0.0

            from ayase.models import QualityMetrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.motion_ac_score = score

            if score == 0.0:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Motion-text mismatch: video is '{predicted}' (flow={mean_flow:.1f}) but caption implies '{expected}'",
                        details={
                            "predicted_motion": predicted,
                            "expected_motion": expected,
                            "mean_optical_flow": float(mean_flow),
                        },
                    )
                )

        except Exception as e:
            logger.warning(f"Motion amplitude analysis failed: {e}")

        return sample

    def _compute_mean_flow(self, sample: Sample) -> Optional[float]:
        """Compute mean RAFT optical flow magnitude across all consecutive pairs.

        motion_ac_score follows EvalCrafter, whose amplitude threshold is
        calibrated to RAFT flow magnitudes; there is no Farneback fallback
        because its differently-scaled flow would misclassify against that
        threshold.
        """
        return self._compute_mean_flow_raft(sample)

    def _compute_mean_flow_raft(self, sample: Sample) -> Optional[float]:
        import torch

        frames = self._load_all_frames(sample)
        if len(frames) < 2:
            return None

        optical_flows = []
        with torch.no_grad():
            for i in range(len(frames) - 1):
                img1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float().unsqueeze(0).to(self._device)
                img2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).float().unsqueeze(0).to(self._device)

                # EvalCrafter: InputPadder (replicate pad to x8), no resize;
                # torchvision RAFT expects inputs normalized to [-1, 1].
                img1 = _pad_to_multiple_of_8(img1 / 255.0 * 2.0 - 1.0)
                img2 = _pad_to_multiple_of_8(img2 / 255.0 * 2.0 - 1.0)

                flow = self._model(img1, img2, num_flow_updates=20)[-1]
                flow_magnitude = torch.norm(flow.squeeze(0), dim=0)
                optical_flows.append(flow_magnitude.mean().item())

        if not optical_flows:
            return None
        return float(np.mean(optical_flows))

    @staticmethod
    def _classify_caption_motion(caption: str) -> Optional[str]:
        caption_lower = caption.lower()
        has_fast = any(kw in caption_lower for kw in FAST_KEYWORDS)
        has_slow = any(kw in caption_lower for kw in SLOW_KEYWORDS)

        if has_fast and not has_slow:
            return "large"
        if has_slow and not has_fast:
            return "slow"
        return None

    def _load_all_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
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
                # EvalCrafter protocol: every frame, full resolution unless the
                # caller explicitly configured a memory cap.
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(_cap_frame_resolution(frame, self.max_resolution))
            cap.release()
        except Exception as e:
            logger.debug(f"Failed to load frames: {e}")
        return frames

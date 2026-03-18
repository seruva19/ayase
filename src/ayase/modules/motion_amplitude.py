"""Motion Amplitude Classification (motion_ac_score).

Matches EvalCrafter implementation: computes average RAFT optical flow
magnitude across all consecutive frame pairs, classifies as "large" (>5)
or "slow" (<=5), and compares against expected motion from the caption.

Score is binary: 100 if match, 0 if mismatch (then *100 for EvalCrafter scale).
"""

import logging
from typing import Optional, List

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

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
    description = "Motion amplitude classification vs caption (motion_ac_score via RAFT)"

    default_config = {
        "amplitude_threshold": 5.0,
        "max_frames": 150,
        "scoring_mode": "binary",  # "binary" (0/100 match) or "continuous" (smooth 0-100)
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.amplitude_threshold = self.config.get("amplitude_threshold", 5.0)
        self.max_frames = self.config.get("max_frames", 150)
        self._model = None
        self._device = "cpu"
        self._ml_available = False
        self._transforms = None

    def setup(self) -> None:
        try:
            import os
            import torch
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

            # Redirect torch hub cache to models_dir so RAFT weights respect config
            models_dir = self.config.get("models_dir")
            if models_dir:
                os.environ["TORCH_HOME"] = str(models_dir)

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Setting up RAFT-Small for motion amplitude on {self._device}...")

            weights = Raft_Small_Weights.DEFAULT
            self._model = raft_small(weights=weights, progress=False).to(self._device)
            self._model.eval()
            self._transforms = weights.transforms()
            self._ml_available = True

        except ImportError:
            logger.warning("torchvision >= 0.13 required for RAFT.")
        except Exception as e:
            logger.warning(f"Failed to setup RAFT: {e}")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
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

            scoring_mode = self.config.get("scoring_mode", "binary")
            if scoring_mode == "continuous":
                # Continuous score: how well flow matches expected amplitude
                if expected in ("large", "fast"):
                    # Higher flow = better match; sigmoid-like scaling
                    score = min(100.0, abs(mean_flow) / self.amplitude_threshold * 50.0)
                else:
                    # Lower flow = better match
                    score = max(0.0, 100.0 - abs(mean_flow) / self.amplitude_threshold * 50.0)
            else:
                score = 100.0 if predicted == expected else 0.0

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
        """Compute mean optical flow magnitude across all consecutive pairs."""
        if self._ml_available:
            return self._compute_mean_flow_raft(sample)
        return self._compute_mean_flow_farneback(sample)

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

                if self._transforms:
                    img1, img2 = self._transforms(img1, img2)

                flow = self._model(img1, img2)[-1]
                flow_magnitude = torch.norm(flow.squeeze(0), dim=0)
                optical_flows.append(flow_magnitude.mean().item())

        if not optical_flows:
            return None
        return float(np.mean(optical_flows))

    def _compute_mean_flow_farneback(self, sample: Sample) -> Optional[float]:
        """Fallback when RAFT is unavailable."""
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return None

        flow_magnitudes = []
        prev_gray = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_magnitudes.append(float(np.mean(mag)))

            prev_gray = gray

        cap.release()

        if not flow_magnitudes:
            return None
        return float(np.mean(flow_magnitudes))

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
                        frames.append(frame)
                    frame_idx += 1
            else:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            cap.release()
        except Exception as e:
            logger.debug(f"Failed to load frames: {e}")
        return frames

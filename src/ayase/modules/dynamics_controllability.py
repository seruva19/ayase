"""Dynamics Controllability module.

Assesses if motion in video matches the dynamics implied by the text prompt.
Parses caption for motion keywords and compares to actual motion.
Range: 0-1 (higher = better controllability/alignment).

Backend tiers:
  1. **CoTracker** — Camera motion classification + object trajectory analysis
  2. **Farneback** — OpenCV dense optical flow (current approach)
  3. **Keyword** — Pure text keyword matching
"""

import logging
import re

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Camera motion keywords mapped to expected motion vectors
CAMERA_KEYWORDS = {
    "pan left": "pan_left", "pan right": "pan_right",
    "panning left": "pan_left", "panning right": "pan_right",
    "tilt up": "tilt_up", "tilt down": "tilt_down",
    "tilting up": "tilt_up", "tilting down": "tilt_down",
    "zoom in": "zoom_in", "zoom out": "zoom_out",
    "zooming in": "zoom_in", "zooming out": "zoom_out",
    "orbit": "orbit", "orbiting": "orbit",
    "tracking shot": "tracking", "dolly": "tracking",
    "static camera": "static", "fixed camera": "static",
    "handheld": "handheld", "shaky": "handheld",
}


class DynamicsControllabilityModule(PipelineModule):
    name = "dynamics_controllability"
    description = "Assesses motion controllability based on text-motion alignment"
    default_config = {
        "subsample": 16,
    }

    # Motion keywords and their expected motion levels (0-1)
    MOTION_KEYWORDS = {
        "static": 0.0, "still": 0.0, "stationary": 0.0, "motionless": 0.0,
        "slow": 0.2, "slowly": 0.2, "gentle": 0.2, "calm": 0.2,
        "moving": 0.5, "walk": 0.5, "walking": 0.5, "moderate": 0.5,
        "fast": 0.8, "quickly": 0.8, "rapid": 0.8, "swift": 0.8,
        "run": 0.8, "running": 0.8, "sprint": 0.9, "dash": 0.9, "race": 0.9,
        "dynamic": 0.7, "energetic": 0.7, "action": 0.8,
        "explosive": 0.9, "sudden": 0.9,
        "smooth": 0.4, "fluid": 0.4, "flowing": 0.4,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = "farneback"
        self._cotracker = None
        self._device = None

    def setup(self) -> None:
        # Tier 1: CoTracker
        try:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._cotracker = torch.hub.load(
                "facebookresearch/co-tracker", "cotracker2"
            ).to(device).eval()
            self._device = device
            self._backend = "cotracker"
            logger.info("DynamicsControllability loaded CoTracker on %s", device)
            return
        except Exception as e:
            logger.info("CoTracker unavailable for dynamics: %s", e)

        # Tier 2: Farneback
        try:
            cv2.calcOpticalFlowFarneback  # noqa: B018
            self._backend = "farneback"
            logger.info("DynamicsControllability using Farneback optical flow")
        except AttributeError:
            logger.info("DynamicsControllability using keyword-only mode")

    def _extract_expected_motion(self, caption: str) -> float:
        caption_lower = caption.lower()
        matched_levels = []
        for keyword, level in self.MOTION_KEYWORDS.items():
            if re.search(r'\b' + keyword + r'\b', caption_lower):
                matched_levels.append(level)
        if not matched_levels:
            return 0.5
        return float(np.mean(matched_levels))

    def _extract_camera_keywords(self, caption: str) -> list:
        caption_lower = caption.lower()
        found = []
        for phrase, motion_type in CAMERA_KEYWORDS.items():
            if phrase in caption_lower:
                found.append(motion_type)
        return found

    def _classify_camera_motion(self, flow: np.ndarray) -> str:
        """Classify global camera motion from dense optical flow field."""
        mean_dx = float(np.mean(flow[..., 0]))
        mean_dy = float(np.mean(flow[..., 1]))

        h, w = flow.shape[:2]
        cy, cx = h / 2, w / 2
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        # Radial flow for zoom detection
        radial_x = flow[..., 0] - mean_dx
        radial_y = flow[..., 1] - mean_dy
        dir_x = (x_coords - cx) / (w / 2 + 1e-6)
        dir_y = (y_coords - cy) / (h / 2 + 1e-6)
        radial_component = radial_x * dir_x + radial_y * dir_y
        mean_radial = float(np.mean(radial_component))

        mag = np.sqrt(mean_dx ** 2 + mean_dy ** 2)
        if mag < 0.5 and abs(mean_radial) < 0.3:
            return "static"
        if mean_radial > 1.0:
            return "zoom_in"
        if mean_radial < -1.0:
            return "zoom_out"
        if abs(mean_dx) > abs(mean_dy) * 1.5:
            return "pan_left" if mean_dx < 0 else "pan_right"
        if abs(mean_dy) > abs(mean_dx) * 1.5:
            return "tilt_up" if mean_dy < 0 else "tilt_down"
        return "tracking"

    def _compute_cotracker(self, sample: Sample) -> tuple:
        """Returns (actual_motion_level, detected_camera_motion)."""
        import torch

        num_frames = self.config.get("subsample", 16)
        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = list(range(0, total, max(1, total // num_frames)))[:num_frames]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(frames) < 3:
            return None, "unknown"

        h, w = frames[0].shape[:2]
        target_h, target_w = min(h, 384), min(w, 512)
        resized = [cv2.resize(f, (target_w, target_h)) for f in frames]

        video_tensor = (
            torch.from_numpy(np.stack(resized))
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .float()
            .to(self._device)
        )

        with torch.no_grad():
            pred_tracks, pred_visibility = self._cotracker(video_tensor)

        tracks = pred_tracks[0].cpu().numpy()  # [T, N, 2]
        visibility = pred_visibility[0].cpu().numpy()

        good = visibility.sum(axis=0) >= len(frames) * 0.5
        if good.sum() < 5:
            return None, "unknown"
        tracks = tracks[:, good, :]

        velocities = np.diff(tracks, axis=0)  # [T-1, N, 2]

        # Global motion (camera) = median velocity per frame
        global_vel = np.median(velocities, axis=1)  # [T-1, 2]

        # Classify camera motion from global velocity
        mean_gx = float(np.mean(global_vel[:, 0]))
        mean_gy = float(np.mean(global_vel[:, 1]))
        global_mag = np.sqrt(mean_gx ** 2 + mean_gy ** 2)

        if global_mag < 1.0:
            camera_motion = "static"
        elif abs(mean_gx) > abs(mean_gy) * 1.5:
            camera_motion = "pan_left" if mean_gx < 0 else "pan_right"
        elif abs(mean_gy) > abs(mean_gx) * 1.5:
            camera_motion = "tilt_up" if mean_gy < 0 else "tilt_down"
        else:
            camera_motion = "tracking"

        # Object motion = velocity after subtracting global
        object_vel = velocities - global_vel[:, np.newaxis, :]
        object_mag = np.sqrt(np.sum(object_vel ** 2, axis=-1))
        total_mag = np.sqrt(np.sum(velocities ** 2, axis=-1))

        avg_motion = float(np.mean(total_mag))
        normalized = min(avg_motion / 15.0, 1.0)

        return normalized, camera_motion

    def _compute_farneback(self, video_path) -> float:
        try:
            cap = cv2.VideoCapture(str(video_path))
            motion_magnitudes = []
            prev_frame = None
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_frame is not None and frame_count % 3 == 0:
                    try:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        motion_magnitudes.append(magnitude.mean())
                    except Exception:
                        pass

                prev_frame = gray
                frame_count += 1

            cap.release()

            if not motion_magnitudes:
                return None

            avg_motion = np.mean(motion_magnitudes)
            normalized_motion = min(avg_motion / 10.0, 1.0)
            return float(normalized_motion)

        except Exception as e:
            logger.debug(f"Farneback motion computation failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample
        if sample.caption is None or not sample.caption.text:
            return sample

        try:
            caption = sample.caption.text
            expected_motion = self._extract_expected_motion(caption)
            camera_keywords = self._extract_camera_keywords(caption)

            actual_motion = None
            camera_motion_match = 1.0

            if self._backend == "cotracker":
                try:
                    actual_motion, detected_camera = self._compute_cotracker(sample)
                    # Camera motion keyword match bonus
                    if camera_keywords and detected_camera != "unknown":
                        if detected_camera in camera_keywords:
                            camera_motion_match = 1.0
                        elif "static" in camera_keywords and detected_camera == "static":
                            camera_motion_match = 1.0
                        else:
                            camera_motion_match = 0.5
                except Exception as e:
                    logger.info("CoTracker dynamics failed: %s, using Farneback", e)

            if actual_motion is None and self._backend in ("farneback", "cotracker"):
                actual_motion = self._compute_farneback(sample.path)

            if actual_motion is None:
                # Cannot compute actual motion — skip rather than storing a meaningless score
                return sample

            # Compute controllability
            error = abs(expected_motion - actual_motion)
            base_controllability = 1.0 - error

            # Blend with camera motion match if camera keywords present
            if camera_keywords:
                controllability = 0.7 * base_controllability + 0.3 * camera_motion_match
            else:
                controllability = base_controllability

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.dynamics_controllability = float(
                np.clip(controllability, 0.0, 1.0)
            )

            logger.debug(
                "Dynamics controllability for %s: %.3f (expected: %.2f, actual: %.2f)",
                sample.path.name, controllability, expected_motion, actual_motion,
            )

        except Exception as e:
            logger.warning("Dynamics controllability processing failed for %s: %s", sample.path, e)

        return sample

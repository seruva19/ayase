"""TRAJAN — trajectory-based motion consistency assessment.

Tracks feature points across video frames and measures trajectory
smoothness to assess motion quality.

Backend tiers:
  1. **CoTracker** — Facebook's dense point tracking model
     (``pip install cotracker``, ``github.com/facebookresearch/co-tracker``)
  2. **Lucas-Kanade** — OpenCV sparse optical flow tracking (heuristic)
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TRAJANModule(PipelineModule):
    name = "trajan"
    description = "Motion consistency via point tracking (CoTracker or Lucas-Kanade fallback)"
    default_config = {
        "num_frames": 16,
        "num_points": 256,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = "lk"
        self._cotracker = None
        self._device = None

    def setup(self) -> None:
        # Tier 1: CoTracker from Facebook Research
        try:
            import torch
            import cotracker

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Load CoTracker model
            self._cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device).eval()
            self._device = device
            self._backend = "cotracker"
            self._ml_available = True
            logger.info("TRAJAN loaded CoTracker on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.info("CoTracker unavailable: %s", e)

        # Tier 1b: Try torch.hub directly without cotracker package
        try:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device).eval()
            self._device = device
            self._backend = "cotracker"
            self._ml_available = True
            logger.info("TRAJAN loaded CoTracker via torch.hub on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.info("CoTracker torch.hub unavailable: %s", e)

        # Tier 2: Lucas-Kanade (OpenCV)
        try:
            import cv2
            self._ml_available = True
            self._backend = "lk"
            logger.info("TRAJAN using Lucas-Kanade optical flow tracking")
        except ImportError:
            logger.warning("TRAJAN unavailable: OpenCV not installed")

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        if not sample.is_video:
            return sample

        try:
            score = self._compute_trajectory_consistency(sample)
            if score is not None:
                sample.quality_metrics.trajan_score = score
        except Exception as e:
            logger.warning("TRAJAN processing failed: %s", e)
        return sample

    def _compute_trajectory_consistency(self, sample: Sample) -> Optional[float]:
        """Compute trajectory consistency using the best available backend."""
        if self._backend == "cotracker":
            try:
                return self._compute_cotracker(sample)
            except Exception as e:
                logger.info("CoTracker failed: %s, falling back to LK", e)

        return self._compute_lk(sample)

    def _compute_cotracker(self, sample: Sample) -> Optional[float]:
        """Compute trajectory consistency using CoTracker."""
        import torch
        import cv2

        num_frames = self.config.get("num_frames", 16)
        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = list(range(0, total, max(1, total // num_frames)))[:num_frames]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
        cap.release()

        if len(frames) < 3:
            return None

        # Prepare video tensor: [1, T, 3, H, W]
        # Resize to manageable size for CoTracker
        h, w = frames[0].shape[:2]
        target_h = min(h, 384)
        target_w = min(w, 512)
        resized = [cv2.resize(f, (target_w, target_h)) for f in frames]

        video_tensor = torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).unsqueeze(0).float().to(self._device)

        with torch.no_grad():
            # CoTracker returns (predictions, visibility)
            # predictions shape: [1, T, N, 2] — N tracked points across T frames
            pred_tracks, pred_visibility = self._cotracker(video_tensor)

        tracks = pred_tracks[0].cpu().numpy()  # [T, N, 2]
        visibility = pred_visibility[0].cpu().numpy()  # [T, N]

        # Filter to well-tracked points (visible in most frames)
        min_visible = len(frames) * 0.6
        good_points = visibility.sum(axis=0) >= min_visible
        if good_points.sum() < 5:
            return None

        tracks = tracks[:, good_points, :]  # [T, N_good, 2]

        # Compute trajectory smoothness from CoTracker output
        velocities = np.diff(tracks, axis=0)  # [T-1, N, 2]
        accelerations = np.diff(velocities, axis=0)  # [T-2, N, 2]

        accel_mag = np.sqrt(np.sum(accelerations ** 2, axis=-1))
        vel_mag = np.sqrt(np.sum(velocities[:-1] ** 2, axis=-1))
        vel_mag = np.maximum(vel_mag, 1e-6)

        jerk_ratio = accel_mag / vel_mag
        mean_jerk = float(np.mean(jerk_ratio))
        smoothness = 1.0 / (1.0 + mean_jerk)

        return float(smoothness)

    def _compute_lk(self, sample: Sample) -> Optional[float]:
        """Compute trajectory consistency using Lucas-Kanade optical flow."""
        import cv2

        num_frames = self.config.get("num_frames", 16)
        num_points = self.config.get("num_points", 256)

        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = list(range(0, total, max(1, total // num_frames)))[:num_frames]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
        cap.release()

        if len(frames) < 3:
            return None

        feature_params = dict(maxCorners=num_points, qualityLevel=0.01, minDistance=10)
        points = cv2.goodFeaturesToTrack(frames[0], mask=None, **feature_params)
        if points is None or len(points) < 10:
            return None

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        trajectories = [points.reshape(-1, 2)]
        current_points = points

        for i in range(1, len(frames)):
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                frames[i - 1], frames[i], current_points, None, **lk_params
            )
            if next_points is None:
                break
            good = status.ravel() == 1
            if good.sum() < 5:
                break
            trajectories.append(next_points[good].reshape(-1, 2))
            current_points = next_points[good].reshape(-1, 1, 2)

        if len(trajectories) < 3:
            return None

        min_pts = min(len(t) for t in trajectories)
        if min_pts < 5:
            return None

        trajectories = [t[:min_pts] for t in trajectories]
        traj_array = np.array(trajectories)  # (T, N, 2)

        velocities = np.diff(traj_array, axis=0)
        accelerations = np.diff(velocities, axis=0)

        accel_mag = np.sqrt(np.sum(accelerations ** 2, axis=-1))
        vel_mag = np.sqrt(np.sum(velocities[:-1] ** 2, axis=-1))
        vel_mag = np.maximum(vel_mag, 1e-6)

        jerk_ratio = accel_mag / vel_mag
        mean_jerk = float(np.mean(jerk_ratio))
        smoothness = 1.0 / (1.0 + mean_jerk)

        return float(smoothness)

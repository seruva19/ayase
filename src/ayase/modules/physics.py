"""Physics plausibility module — VBench-2.0 dimension.

Tracks keypoints across video frames and analyzes trajectories for
physically implausible motion (teleportation, impossible acceleration,
gravity violations).

Backend tiers:
  1. **CoTracker** — Facebook's dense point tracking model
  2. **Lucas-Kanade** — OpenCV sparse optical flow tracking
  3. **Heuristic** — Frame differencing + pixel acceleration
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PhysicsModule(PipelineModule):
    name = "physics"
    description = "Physics plausibility via trajectory analysis (CoTracker / LK / heuristic)"
    default_config = {
        "subsample": 16,
        "accel_threshold": 50.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = "heuristic"
        self._ml_available = False
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
            self._ml_available = True
            logger.info("Physics loaded CoTracker on %s", device)
            return
        except Exception as e:
            logger.info("CoTracker unavailable for physics: %s", e)

        # Tier 2: Lucas-Kanade (OpenCV)
        try:
            cv2.calcOpticalFlowPyrLK  # noqa: B018
            self._backend = "lk"
            self._ml_available = True
            logger.info("Physics using Lucas-Kanade optical flow")
        except AttributeError:
            logger.info("Physics falling back to heuristic")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            score = self._compute_physics_score(sample)
            if score is not None:
                sample.quality_metrics.physics_score = score
                # Preserve backward-compatible validation issue
                if score < 0.5:
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Physically implausible motion (physics_score={score:.2f})",
                            details={"physics_score": score, "backend": self._backend},
                        )
                    )
        except Exception as e:
            logger.warning("Physics processing failed: %s", e)

        return sample

    def _compute_physics_score(self, sample: Sample) -> Optional[float]:
        if self._backend == "cotracker":
            try:
                return self._compute_cotracker(sample)
            except Exception as e:
                logger.info("CoTracker failed for physics: %s, falling back to LK", e)

        if self._backend in ("lk", "cotracker"):
            try:
                return self._compute_lk(sample)
            except Exception as e:
                logger.info("LK failed for physics: %s, falling back to heuristic", e)

        return self._compute_heuristic(sample)

    # ------------------------------------------------------------------ #
    # Tier 1: CoTracker                                                    #
    # ------------------------------------------------------------------ #

    def _compute_cotracker(self, sample: Sample) -> Optional[float]:
        import torch

        num_frames = self.config.get("subsample", 16)
        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 10:
            cap.release()
            return None

        indices = list(range(0, total, max(1, total // num_frames)))[:num_frames]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(frames) < 5:
            return None

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
        visibility = pred_visibility[0].cpu().numpy()  # [T, N]

        good_points = visibility.sum(axis=0) >= len(frames) * 0.6
        if good_points.sum() < 5:
            return None

        tracks = tracks[:, good_points, :]
        return self._score_from_tracks(tracks)

    # ------------------------------------------------------------------ #
    # Tier 2: Lucas-Kanade                                                 #
    # ------------------------------------------------------------------ #

    def _compute_lk(self, sample: Sample) -> Optional[float]:
        num_frames = self.config.get("subsample", 16)
        accel_threshold = self.config.get("accel_threshold", 50.0)

        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 10:
            cap.release()
            return None

        indices = list(range(0, total, max(1, total // num_frames)))[:num_frames]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()

        if len(frames) < 5:
            return None

        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        p0 = cv2.goodFeaturesToTrack(frames[0], mask=None, **feature_params)
        if p0 is None or len(p0) < 5:
            return None

        trajectories = [p0.reshape(-1, 2)]
        current = p0

        for i in range(1, len(frames)):
            p1, st, _ = cv2.calcOpticalFlowPyrLK(frames[i - 1], frames[i], current, None, **lk_params)
            if p1 is None:
                break
            good = st.ravel() == 1
            if good.sum() < 5:
                break
            trajectories.append(p1[good].reshape(-1, 2))
            current = p1[good].reshape(-1, 1, 2)

        if len(trajectories) < 4:
            return None

        min_pts = min(len(t) for t in trajectories)
        if min_pts < 5:
            return None
        trajectories = [t[:min_pts] for t in trajectories]
        tracks = np.array(trajectories)  # [T, N, 2]
        return self._score_from_tracks(tracks, accel_threshold)

    # ------------------------------------------------------------------ #
    # Tier 3: Heuristic (frame differencing)                               #
    # ------------------------------------------------------------------ #

    def _compute_heuristic(self, sample: Sample) -> Optional[float]:
        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 10:
            cap.release()
            return None

        num_frames = min(self.config.get("subsample", 16), total)
        indices = list(range(0, total, max(1, total // num_frames)))[:num_frames]

        diffs = []
        prev_gray = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_gray is not None:
                diff = np.abs(gray - prev_gray).mean()
                diffs.append(diff)
            prev_gray = gray
        cap.release()

        if len(diffs) < 3:
            return None

        diffs = np.array(diffs)
        velocities = diffs
        accelerations = np.abs(np.diff(velocities))

        if len(accelerations) == 0:
            return 1.0

        # Fraction of frames with plausible acceleration
        threshold = self.config.get("accel_threshold", 50.0) * 0.2  # scaled for pixel diffs
        teleport_frac = float(np.mean(accelerations > threshold))
        gravity_score = 1.0 - teleport_frac

        # Smoothness of velocity changes
        if velocities.std() > 1e-6:
            cv = float(accelerations.std() / (velocities.mean() + 1e-6))
            smoothness = 1.0 / (1.0 + cv)
        else:
            smoothness = 1.0

        score = 0.5 * gravity_score + 0.5 * smoothness
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Shared scoring                                                       #
    # ------------------------------------------------------------------ #

    def _score_from_tracks(self, tracks: np.ndarray, accel_threshold: float = 50.0) -> float:
        """Score physics plausibility from tracked points [T, N, 2]."""
        velocities = np.diff(tracks, axis=0)  # [T-1, N, 2]
        accelerations = np.diff(velocities, axis=0)  # [T-2, N, 2]

        if len(accelerations) == 0:
            return 1.0

        accel_mag = np.sqrt(np.sum(accelerations ** 2, axis=-1))  # [T-2, N]
        vel_mag = np.sqrt(np.sum(velocities[:-1] ** 2, axis=-1))  # [T-2, N]

        # Gravity score: fraction of points with plausible vertical acceleration
        vert_accel = np.abs(accelerations[..., 1])  # vertical component
        vert_plausible = vert_accel < accel_threshold
        gravity_score = float(np.mean(vert_plausible))

        # Teleport score: fraction of impossible accelerations
        teleport_frac = float(np.mean(accel_mag > accel_threshold))
        teleport_score = 1.0 - teleport_frac

        # Collision score: velocity discontinuities that are physically plausible
        vel_mag_safe = np.maximum(vel_mag, 1e-6)
        jerk_ratio = accel_mag / vel_mag_safe
        collision_score = 1.0 / (1.0 + float(np.mean(jerk_ratio)))

        score = (gravity_score + collision_score + teleport_score) / 3.0
        return float(np.clip(score, 0.0, 1.0))

"""TRAJAN — trajectory-based motion consistency assessment.

Tracks dense feature points across video frames with Facebook Research's
CoTracker model and measures trajectory smoothness (jerk relative to
velocity) to assess motion consistency.

Backend:
  * **CoTracker** — dense point tracking model
    (``github.com/facebookresearch/co-tracker``, loaded via ``torch.hub``).

CoTracker is the only backend: if it is unavailable the module reports
itself unavailable rather than substituting an OpenCV Lucas-Kanade
optical-flow proxy.
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
    metric_groups = {
        "trajan_score": "motion",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._cotracker = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import torch
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._cotracker = (
                torch.hub.load("facebookresearch/co-tracker", "cotracker2")
                .to(self._device)
                .eval()
            )
            self._backend = "cotracker"
            self._ml_available = True
            logger.info("TRAJAN loaded CoTracker on %s", self._device)
            return
        except ImportError as e:
            logger.info("CoTracker unavailable (missing dependency): %s", e)
        except Exception as e:
            logger.info("CoTracker unavailable: %s", e)

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "TRAJAN unavailable: CoTracker could not be loaded; trajan_score will "
            "not be populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        if not sample.is_video:
            return sample

        try:
            score = self._compute_cotracker(sample)
            if score is not None:
                sample.quality_metrics.trajan_score = score
        except Exception as e:
            logger.warning("TRAJAN processing failed: %s", e)
        return sample

    def _compute_cotracker(self, sample: Sample) -> Optional[float]:
        """Compute trajectory consistency using CoTracker."""
        import torch
        import cv2

        num_frames = self.config.get("num_frames", 16)
        cap = cv2.VideoCapture(str(sample.path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // num_frames)))[:num_frames]

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb)
        finally:
            cap.release()

        if len(frames) < 3:
            return None

        # Resize to a manageable size for CoTracker: [1, T, 3, H, W]
        h, w = frames[0].shape[:2]
        target_h = min(h, 384)
        target_w = min(w, 512)
        resized = [cv2.resize(f, (target_w, target_h)) for f in frames]

        video_tensor = (
            torch.from_numpy(np.stack(resized))
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .float()
            .to(self._device)
        )

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

        # Trajectory smoothness from CoTracker output
        velocities = np.diff(tracks, axis=0)  # [T-1, N, 2]
        accelerations = np.diff(velocities, axis=0)  # [T-2, N, 2]

        accel_mag = np.sqrt(np.sum(accelerations ** 2, axis=-1))
        vel_mag = np.sqrt(np.sum(velocities[:-1] ** 2, axis=-1))
        vel_mag = np.maximum(vel_mag, 1e-6)

        jerk_ratio = accel_mag / vel_mag
        mean_jerk = float(np.mean(jerk_ratio))
        smoothness = 1.0 / (1.0 + mean_jerk)

        return float(smoothness)

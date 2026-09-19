"""Official Fréchet Video Motion Distance for two video distributions.

FVMD (Liu et al., 2024; arXiv:2407.16124) measures motion-consistency
distribution shift. PIPs++ tracks a fixed 20×20 grid in overlapping 16-frame,
256×256 clips; the released evaluator encodes velocity and its published
``acceleration`` channel as local magnitude-weighted orientation histograms,
then applies Fréchet distance. Lower is better and zero denotes identical
empirical feature distributions.

This is a dataset-level metric: it requires at least two generated videos and
two reference videos, with enough frames to produce 16-frame clips. It is not a
paired-video fidelity score, an identity metric, or evidence of a person's
motion manner. Scores depend on frame rate, preprocessing, sample composition,
and extracted-window count; compare datasets prepared under one protocol.

Ayase follows the official Apache-2.0 evaluator at pinned commit
``875a86a92239e5f5751fda52c30d18a062fcfebc``. That release computes its
acceleration channel from trajectories with two leading zero fields, although
paper Eq. (2) specifies a second difference of velocity. Ayase preserves the
released behavior for numerical compatibility and records the backend as
``official_fvmd_1_0_0_compat`` rather than silently correcting the formula.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.config import download_model_file
from ayase.models import Sample

logger = logging.getLogger(__name__)

PIPS_WEIGHTS_URL = (
    "https://github.com/ljh0v0/FVMD-frechet-video-motion-distance/"
    "releases/download/pips2_weights/pips2_weights.pth"
)
PIPS_WEIGHTS_SHA256 = "74dd05ba7f85a34784851f50fede610a99f9acdeb20620b134596837021de3d1"


class FVMDModule(BatchMetricModule):
    """Compare generated/reference motion distributions with official FVMD."""

    name = "fvmd"
    description = "Official PIPs++ velocity/acceleration-histogram Fréchet distance"
    default_config = {
        "models_dir": "models",
        "device": "auto",
        "window_stride": 1,
        "max_windows_per_video": None,
        "subsample_videos": None,
    }
    models = [
        {
            "id": "fvmd/pips2_weights.pth",
            "type": "local",
            "url": PIPS_WEIGHTS_URL,
            "task": "Official FVMD PIPs++ point tracker",
            "size": "421,766,633 bytes",
            "vram": "~511 MiB measured for two 16-frame windows on H100",
            "auto_download": "yes",
            "notes": "Official FVMD 1.0.0 release checkpoint; SHA-256 pinned in module",
        }
    ]
    metric_info = {
        "fvmd": "Official FVMD 1.0.0-compatible dataset distance (lower=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.models_dir = str(self.config.get("models_dir", "models"))
        self.device_config = self.config.get("device", "auto")
        self.window_stride = max(1, int(self.config.get("window_stride", 1)))
        max_windows = self.config.get("max_windows_per_video")
        self.max_windows_per_video = None if max_windows is None else max(1, int(max_windows))
        subsample = self.config.get("subsample_videos")
        self.subsample_videos = None if subsample is None else max(1, int(subsample))
        self.device = None
        self._model = None
        self._ml_available = False
        self._backend = "unavailable"
        self._processed_count = 0

    def setup(self) -> None:
        self._processed_count = 0
        try:
            import torch

            from ayase.runtime import resolve_torch_device, shared_runtime_resource
            from ayase.vendor.fvmd_official import load_pips

            self.device = torch.device(resolve_torch_device(self.device_config))
            checkpoint = download_model_file(
                "fvmd/pips2_weights.pth", PIPS_WEIGHTS_URL, self.models_dir
            )
            digest = self._sha256(checkpoint)
            if digest != PIPS_WEIGHTS_SHA256:
                raise ValueError(
                    "FVMD checkpoint SHA-256 mismatch: "
                    f"expected {PIPS_WEIGHTS_SHA256}, got {digest}"
                )

            def load_model():
                # The official checkpoint contains legacy pickle globals and
                # cannot use torch's weights_only loader. Deserialization is
                # allowed only after the pinned SHA-256 check above succeeds.
                return load_pips(
                    checkpoint,
                    device=self.device,
                    progress=False,
                    trusted_checkpoint=True,
                )

            self._model = shared_runtime_resource(
                self,
                ("fvmd_pips2", PIPS_WEIGHTS_SHA256, str(self.device)),
                load_model,
            )
            self._ml_available = True
            self._backend = "official_fvmd_1_0_0_compat"
        except ImportError as exc:
            logger.warning("FVMD dependencies are unavailable: %s", exc)
        except Exception as exc:  # noqa: BLE001 - optional metric degrades gracefully
            logger.warning("FVMD setup failed: %s", exc)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract one official 1024-D motion feature per sliding video window."""
        if not self._ml_available or not sample.is_video:
            return None
        if self.subsample_videos is not None and self._processed_count >= self.subsample_videos:
            return None
        try:
            frames = self._decode_frames(Path(sample.path))
            if frames is None or frames.shape[0] < 16:
                return None
            features = self._extract_window_features(frames)
            if features.shape[0] == 0:
                return None
            self._processed_count += 1
            return features
        except Exception as exc:  # noqa: BLE001 - sample failure must not abort pipeline
            logger.warning("FVMD feature extraction failed for %s: %s", sample.path, exc)
            return None

    def extract_reference_features(self, sample: Sample) -> Optional[np.ndarray]:
        previous = self._processed_count
        try:
            return self.extract_features(sample)
        finally:
            self._processed_count = previous

    @staticmethod
    def _decode_frames(path: Path) -> Optional[np.ndarray]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return None
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR))
        finally:
            capture.release()
        return np.asarray(frames, dtype=np.uint8) if frames else None

    def _window_starts(self, frame_count: int) -> np.ndarray:
        starts = np.arange(0, frame_count - 16 + 1, self.window_stride, dtype=np.int64)
        if self.max_windows_per_video is not None and starts.size > self.max_windows_per_video:
            indices = np.linspace(
                0, starts.size - 1, self.max_windows_per_video, dtype=np.int64
            )
            starts = starts[indices]
        return starts

    def _extract_window_features(self, frames: np.ndarray) -> np.ndarray:
        import torch

        from ayase.vendor.fvmd_official import (
            calc_acceleration,
            calc_velocity,
            combine_motion_histograms,
            run_tracking,
        )

        outputs = []
        for start in self._window_starts(frames.shape[0]):
            window = frames[int(start) : int(start) + 16]
            tensor = (
                torch.from_numpy(np.ascontiguousarray(window))
                .permute(0, 3, 1, 2)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                trajectories = run_tracking(self._model, tensor, N=400, iters=16)
                velocity = calc_velocity(trajectories)
                acceleration = calc_acceleration(trajectories)
            feature = combine_motion_histograms(
                velocity.cpu().numpy(), acceleration.cpu().numpy()
            )
            outputs.append(feature[0])
        return np.asarray(outputs, dtype=np.float64)

    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> float:
        """Compute official Fréchet distance; a real reference set is mandatory."""
        if reference_features is None:
            raise ValueError("FVMD requires a separate reference video distribution")
        generated = np.concatenate([np.asarray(item) for item in features], axis=0)
        reference = np.concatenate([np.asarray(item) for item in reference_features], axis=0)
        if generated.shape[0] < 2 or reference.shape[0] < 2:
            raise ValueError("FVMD requires at least two motion windows in each distribution")
        if generated.shape[1:] != reference.shape[1:]:
            raise ValueError("generated/reference FVMD features have different dimensions")
        from ayase.vendor.fvmd_official import calculate_fd_given_vectors

        score = float(calculate_fd_given_vectors(generated, reference))
        if not np.isfinite(score):
            raise ValueError("FVMD produced a non-finite distance")
        return max(0.0, score)

    def on_dispose(self) -> None:
        try:
            super().on_dispose()
        finally:
            self._processed_count = 0


__all__ = ["FVMDModule"]

"""FD_g / FD_k — Frechet distance on gesture poses and kinematic velocities.

Audio2Photoreal (Meta, ICCV 2023) and AI Choreographer (Li et al., SIGGRAPH
2021) evaluate gesture/motion quality with a Frechet distance between the
generated and reference distributions of per-frame pose vectors (FD_g) and
their frame-to-frame velocities (FD_k). Poses are the 33 BlazePose joints
(x, y) from MediaPipe, flattened per frame; velocities are successive
differences.

fd_g / fd_k (dataset-level) -- lower = closer to the reference set.
Requires ``sample.reference_path`` for every sample to build the reference
distribution.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.models import Sample
from .fd_3dmm import _frechet_distance, _gaussian

logger = logging.getLogger(__name__)


class FDGestureModule(BatchMetricModule):
    name = "fd_gk"
    description = "FD_g/FD_k: Frechet on body-pose and velocity distributions vs reference set"
    provenance = "adapted"
    sources = {
        "fd_g": "FD_g, Audio2Photoreal (arXiv:2401.01885); AI Choreographer (arXiv:2101.08779) — https://github.com/facebookresearch/audio2photoreal",
        "fd_k": "FD_k, Audio2Photoreal (arXiv:2401.01885); AI Choreographer (arXiv:2101.08779) — https://github.com/facebookresearch/audio2photoreal",
    }
    deviations = {
        "fd_g": "poses are MediaPipe BlazePose 33-joint xy instead of the source's SMPL/skeleton poses, so absolute values are not comparable",
        "fd_k": "velocities are frame differences of MediaPipe BlazePose 33-joint xy instead of the source's skeleton, so absolute values are not comparable",
    }
    requires_reference = True
    default_config = {
        "fps": 25,
        "max_frames": 600,
    }
    metric_info = {
        "fd_g": "Frechet distance on body-pose distributions vs reference set (lower=closer)",
        "fd_k": "Frechet distance on body-velocity distributions vs reference set (lower=closer)",
    }
    metric_groups = {"fd_g": "motion", "fd_k": "motion"}

    def __init__(self, config=None):
        super().__init__(config)
        self._pose = None
        self._backend = None

    def setup(self) -> None:
        if self.config.get("test_mode"):
            self._backend = "unavailable"
            return
        try:
            import mediapipe as mp

            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=1,
                enable_segmentation=False, min_detection_confidence=0.5,
            )
            self._backend = "mediapipe"
        except ImportError:
            logger.warning("fd_gk: mediapipe not installed, disabled")
            self._backend = "unavailable"
        except Exception as e:
            logger.warning("fd_gk: Pose init failed: %s", e)
            self._backend = "unavailable"

    def on_dispose(self) -> None:
        try:
            if self._pose is not None:
                self._pose.close()
        except Exception:
            pass
        self._pose = None
        if not self._feature_cache or not self._reference_cache:
            self._feature_cache = []
            self._reference_cache = []
            return
        try:
            self.compute_distribution_metric(
                self._feature_cache, self._reference_cache)
        except Exception as e:
            logger.error("fd_gk: dataset FD failed: %s", e)
        finally:
            self._feature_cache = []
            self._reference_cache = []

    def extract_features(self, sample: Sample):
        """(T, 132) pose+velocity feature rows for one video."""
        if self._backend != "mediapipe":
            return None
        if not sample.is_video:
            return None
        from ._mp_seq import body_pose_seq

        res = body_pose_seq(Path(sample.path),
                            fps=float(self.config.get("fps", 25)),
                            max_frames=int(self.config.get("max_frames", 600)),
                            detector=self._pose)
        if res is None or len(res[0]) < 3:
            return None
        xy = res[0][:, :, :2].reshape(len(res[0]), -1)  # [T, 66]
        vel = np.diff(xy, axis=0)                      # [T-1, 66]
        return np.concatenate([xy[1:], vel], axis=1)   # [T-1, 132]

    def process(self, sample: Sample) -> Sample:
        feats = self.extract_features(sample)
        if feats is not None:
            self._feature_cache.append(feats)

        reference_path = getattr(sample, "reference_path", None)
        if reference_path is not None:
            try:
                reference_path = Path(reference_path)
                if reference_path.is_file():
                    ref = Sample(path=reference_path, is_video=sample.is_video)
                    ref_feats = self.extract_features(ref)
                    if ref_feats is not None:
                        self._reference_cache.append(ref_feats)
            except Exception:
                pass
        return sample

    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> Optional[float]:
        if not reference_features:
            return None
        for name, sl in (("fd_g", slice(0, 66)), ("fd_k", slice(66, 132))):
            g = _gaussian(features, sl)
            r = _gaussian(reference_features, sl)
            if g is None or r is None:
                continue
            fd = _frechet_distance(g[0], g[1], r[0], r[1])
            if fd is None or not np.isfinite(fd):
                continue
            if hasattr(self, "pipeline") and self.pipeline and hasattr(
                    self.pipeline, "add_dataset_metric"):
                self.pipeline.add_dataset_metric(name, fd)
        return None


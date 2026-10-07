"""LMD / F-LMD — lip and face landmark distance against the source video.

Chen et al. (ECCV 2018, "Lip Movements Generation") evaluate reenactment with
the mean Euclidean distance between per-frame facial landmarks of the
generated and source videos: LMD on the lip-region landmarks, F-LMD over the
full landmark set. Landmark coordinates are normalised by the per-frame face
bounding-box diagonal for scale invariance.

lmd / f_lmd -- lower = closer to the source (0+).
Requires ``sample.reference_path`` pointing at the source video.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _lip_indices() -> set:
    """MediaPipe FACEMESH_LIPS landmark index set."""
    import mediapipe as mp

    return {i for conn in mp.solutions.face_mesh.FACEMESH_LIPS for i in conn}


def _norm_pts(seq: np.ndarray) -> np.ndarray:
    """xy coords normalised by per-frame face bbox diagonal."""
    xy = seq[..., :2]
    lo = xy.min(axis=1, keepdims=True)
    hi = xy.max(axis=1, keepdims=True)
    diag = np.linalg.norm((hi - lo)[:, 0, :], axis=1, keepdims=True)
    return xy / np.maximum(diag, 1e-6)[:, :, None]


class LMDModule(PipelineModule):
    name = "lmd"
    description = "LMD/F-LMD: landmark distance of lips and full face to the source video"
    provenance = "adapted"
    sources = {
        "lmd": "LMD, Chen et al., ECCV 2018 (arXiv:1803.10404) — https://github.com/lelechen63/ATVGnet",
        "f_lmd": "F-LMD, Chen et al., ECCV 2018 (arXiv:1803.10404) — https://github.com/lelechen63/ATVGnet",
    }
    deviations = {
        "lmd": "landmarks come from MediaPipe FaceMesh (468 pts, lip subset) instead of the source's 68-pt dlib landmarks, and coordinates are bbox-normalised rather than raw pixels",
        "f_lmd": "landmarks come from MediaPipe FaceMesh (468 pts) instead of the source's 68-pt dlib landmarks, and coordinates are bbox-normalised rather than raw pixels",
    }
    requires_reference = True
    default_config = {
        "fps": 25,
        "max_frames": 600,
    }
    metric_info = {
        "lmd": "Mean normalised landmark distance on lip region vs source (lower=better)",
        "f_lmd": "Mean normalised landmark distance over all face landmarks vs source (lower=better)",
    }
    metric_groups = {"lmd": "face", "f_lmd": "face"}

    def __init__(self, config=None):
        super().__init__(config)
        self._mesh = None
        self._backend = None
        self._lips = set()

    def setup(self) -> None:
        if self.config.get("test_mode"):
            self._backend = "unavailable"
            return
        try:
            import mediapipe as mp

            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5,
            )
            self._lips = _lip_indices()
            self._backend = "mediapipe"
        except ImportError:
            logger.warning("lmd: mediapipe not installed, disabled")
            self._backend = "unavailable"
        except Exception as e:
            logger.warning("lmd: FaceMesh init failed: %s", e)
            self._backend = "unavailable"

    def on_dispose(self) -> None:
        if self._mesh is not None:
            try:
                self._mesh.close()
            except Exception:
                pass
            self._mesh = None

    def process(self, sample: Sample) -> Sample:
        if self._backend != "mediapipe":
            return sample
        if not sample.is_video:
            return sample
        ref = sample.reference_path
        if ref is None or not Path(ref).is_file():
            return sample
        try:
            from ._mp_seq import face_mesh_seq

            fps = float(self.config.get("fps", 25))
            max_frames = int(self.config.get("max_frames", 600))
            cand = face_mesh_seq(Path(sample.path), fps=fps,
                                 max_frames=max_frames, mesh=self._mesh)
            src = face_mesh_seq(Path(ref), fps=fps,
                                max_frames=max_frames, mesh=self._mesh)
            if cand is None or src is None:
                return sample
            n = min(len(cand[0]), len(src[0]))
            if n == 0:
                return sample
            c = _norm_pts(cand[0][:n])
            s = _norm_pts(src[0][:n])
            lips = sorted(self._lips)
            lmd = float(np.linalg.norm(c[:, lips] - s[:, lips], axis=-1).mean())
            f_lmd = float(np.linalg.norm(c - s, axis=-1).mean())
            if not np.isfinite(lmd) or not np.isfinite(f_lmd):
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.lmd = lmd
            sample.quality_metrics.f_lmd = f_lmd
        except Exception as e:
            logger.warning("lmd: failed on %s: %s", Path(sample.path).name, e)
        return sample

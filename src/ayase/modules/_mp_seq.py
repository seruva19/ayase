"""Shared per-frame MediaPipe landmark-sequence extraction.

Two self-contained solutions-API extractors (weights bundled with the
mediapipe wheel — no download):

- ``body_pose_seq``: 33 BlazePose landmarks per frame (x, y, z, visibility).
- ``face_mesh_seq``: 468 face-mesh landmarks per frame (x, y, z).

Both return ``(array, frame_indices)`` or None; frames without a detection are
skipped and reported via ``frame_indices``. Used by the gesture/lip evaluation
modules (``lmd``, ``pose_fidelity``, ``beat_consistency``, ``fd_gk``,
``gesture_diversity``).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _iter_frames(video_path: Path, stride: int, max_frames: int):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    idx = -1
    read = 0
    try:
        while read < max_frames:
            ok = cap.grab()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            read += 1
            yield idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _source_stride(video_path: Path, fps: float) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    cap.release()
    return max(1, int(round(float(src_fps) / fps))) if fps > 0 else 1


def body_pose_seq(video_path: Path, fps: float = 0.0, max_frames: int = 600,
                  model_complexity: int = 1, detector=None
                  ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Per-frame 33x4 BlazePose landmarks; uses MediaPipe solutions Pose."""
    import mediapipe as mp

    stride = _source_stride(video_path, fps) if fps > 0 else 1
    pose = detector or mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=model_complexity,
        enable_segmentation=False, min_detection_confidence=0.5,
    )
    close = detector is None
    seq, inds = [], []
    try:
        for idx, rgb in _iter_frames(video_path, stride, max_frames):
            res = pose.process(rgb)
            if res is None or res.pose_landmarks is None:
                continue
            lm = res.pose_landmarks.landmark
            seq.append([[p.x, p.y, p.z, p.visibility] for p in lm])
            inds.append(idx)
    except Exception as e:
        logger.warning("mp_seq: body pose failed on %s: %s", video_path.name, e)
        return None
    finally:
        if close:
            pose.close()
    if not seq:
        return None
    return np.asarray(seq, dtype=np.float32), np.asarray(inds, dtype=np.int64)


def face_mesh_seq(video_path: Path, fps: float = 0.0, max_frames: int = 600,
                  mesh=None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Per-frame 468x3 face-mesh landmarks; uses MediaPipe solutions FaceMesh."""
    import mediapipe as mp

    stride = _source_stride(video_path, fps) if fps > 0 else 1
    face = mesh or mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    close = mesh is None
    seq, inds = [], []
    try:
        for idx, rgb in _iter_frames(video_path, stride, max_frames):
            res = face.process(rgb)
            faces = getattr(res, "multi_face_landmarks", None)
            if not faces:
                continue
            lm = faces[0].landmark
            seq.append([[p.x, p.y, p.z] for p in lm])
            inds.append(idx)
    except Exception as e:
        logger.warning("mp_seq: face mesh failed on %s: %s", video_path.name, e)
        return None
    finally:
        if close:
            face.close()
    if not seq:
        return None
    return np.asarray(seq, dtype=np.float32), np.asarray(inds, dtype=np.int64)


def normalize_xy(pts: np.ndarray) -> np.ndarray:
    """Normalize (..., >=2) xy coords by the bbox diagonal — pose-eval
    convention for scale invariance (FOMM AKD normalisation)."""
    xy = pts[..., :2]
    lo = xy.min(axis=-2, keepdims=True)
    hi = xy.max(axis=-2, keepdims=True)
    diag = np.linalg.norm((hi - lo)[..., 0, :], axis=-1, keepdims=True)
    diag = np.maximum(diag, 1e-6)
    return xy / diag[..., None]

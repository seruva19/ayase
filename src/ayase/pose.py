"""Per-frame human keypoint PRIMITIVE (RTMPose + YOLOX detection).

:mod:`ayase.modules.rtmpose_fidelity` runs a person detector and a pose estimator
over the clip and then reduces everything to a single scalar (``rtmpose_score``).
Consumers that need the underlying skeletons -- comparing a generated clip against
a driving video, measuring contact between two people, following a person across
frames, deciding whether a track ended at the frame border or in mid-frame -- have
no way to obtain them from that module.

This module exposes the raw per-person detections for a single frame: bounding box,
COCO-17 keypoints and per-keypoint confidences, using the same rtmlib backend and
weights as ``rtmpose_fidelity`` so that keypoint-derived metrics stay numerically
consistent with ``rtmpose_score``.

The heavy ``rtmlib`` import happens inside the loader, not at module import time,
so importing :mod:`ayase.pose` stays cheap for consumers that never call it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO-17 keypoint indices, named so that consumers do not hard-code integers.
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

#: Confidence below which a keypoint should be treated as absent. Same value the
#: pose-plausibility module uses, so consumers agree on what "found" means.
KEYPOINT_CONF = 0.3

_MODELS_BASE = "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/"
_DET_REL = "rtmpose_fidelity/yolox_m.onnx"
_POSE_REL = "rtmpose_fidelity/rtmpose_m.onnx"

_DETPOSE_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any]] = {}


def load_pose_backend(
    device: str = "auto",
    models_dir: str = "models",
    det_input_size: Tuple[int, int] = (640, 640),
    pose_input_size: Tuple[int, int] = (192, 256),
) -> Optional[Tuple[Any, Any]]:
    """Return an ``(detector, pose_estimator)`` pair, or ``None`` if unavailable.

    Returns ``None`` rather than raising: a missing optional backend is a normal
    condition for a metrics toolkit, and consumers degrade by leaving their own
    values unset.
    """
    from ayase.config import download_model_file
    from ayase.runtime import resolve_torch_device

    torch_device = resolve_torch_device(device)
    rt_device = "cuda" if "cuda" in str(torch_device) else "cpu"

    # Same relative paths and mirror as rtmpose_fidelity, on purpose: the weights
    # are shared, so a consumer of this primitive never triggers a second download
    # and keypoints stay identical to the ones behind ``rtmpose_score``.
    try:
        det_path = str(download_model_file(_DET_REL, _MODELS_BASE + _DET_REL, models_dir))
        pose_path = str(download_model_file(_POSE_REL, _MODELS_BASE + _POSE_REL, models_dir))
    except Exception as exc:  # pragma: no cover - depends on local model store
        logger.warning("pose primitive: model files unavailable (%s)", exc)
        return None

    key = (str(det_path), str(pose_path), rt_device, det_input_size, pose_input_size)
    cached = _DETPOSE_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from rtmlib import RTMPose, YOLOX
    except ImportError:
        logger.warning("pose primitive: rtmlib not installed; keypoints unavailable")
        return None

    try:
        det = YOLOX(
            onnx_model=str(det_path),
            model_input_size=tuple(det_input_size),
            backend="onnxruntime",
            device=rt_device,
        )
        pose = RTMPose(
            onnx_model=str(pose_path),
            model_input_size=tuple(pose_input_size),
            backend="onnxruntime",
            device=rt_device,
        )
    except Exception as exc:  # pragma: no cover - backend/runtime specific
        logger.warning("pose primitive: backend failed to load (%s)", exc)
        return None

    _DETPOSE_CACHE[key] = (det, pose)
    return det, pose


def pose_keypoints(
    frame_bgr: np.ndarray,
    *,
    device: str = "auto",
    models_dir: str = "models",
    backend: Optional[Tuple[Any, Any]] = None,
) -> List[Dict[str, Any]]:
    """People detected in one BGR frame, largest first.

    Each entry holds ``box`` (x1, y1, x2, y2), ``keypoints`` (17, 2), ``scores``
    (17,) and ``area``. An empty list means no person was detected -- which is a
    result, not a failure.

    Ordering is by bounding-box area, descending. It is deliberate and stable:
    consumers that care about "the subject" take the first entry, and consumers
    that track people across frames need a deterministic order to match against.
    """
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return []

    detpose = backend if backend is not None else load_pose_backend(device, models_dir)
    if detpose is None:
        return []
    detector, estimator = detpose

    # rtmlib/onnxruntime need a writable contiguous array; sampled frames are
    # frequently read-only views.
    image = np.ascontiguousarray(frame_bgr)
    try:
        boxes = detector(image)
    except Exception as exc:  # pragma: no cover - backend specific
        logger.warning("pose primitive: detection failed (%s)", exc)
        return []
    if boxes is None or len(boxes) == 0:
        return []

    try:
        keypoints, scores = estimator(image, boxes)
    except Exception as exc:  # pragma: no cover - backend specific
        logger.warning("pose primitive: pose estimation failed (%s)", exc)
        return []
    if keypoints is None or len(keypoints) == 0:
        return []

    people: List[Dict[str, Any]] = []
    for index, box in enumerate(np.asarray(boxes, dtype=float)):
        if index >= len(keypoints):
            break
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        people.append(
            {
                "box": (x1, y1, x2, y2),
                "keypoints": np.asarray(keypoints[index], dtype=float),
                "scores": np.asarray(scores[index], dtype=float),
                "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
            }
        )
    people.sort(key=lambda person: -person["area"])
    return people


def body_scale(keypoints: np.ndarray, scores: np.ndarray,
               min_conf: float = KEYPOINT_CONF) -> Optional[float]:
    """Scale factor for making two skeletons comparable, or ``None``.

    Shoulder width first, shoulder-to-hip length as a fallback. Hip-based
    normalisation alone is not usable: waist-up framing -- common in talking-head
    and avatar footage -- has no hips in frame at all, and a metric normalised by
    them silently degenerates into a measure of how tightly the shot is cropped.
    """
    if scores[LEFT_SHOULDER] >= min_conf and scores[RIGHT_SHOULDER] >= min_conf:
        width = float(np.linalg.norm(keypoints[LEFT_SHOULDER] - keypoints[RIGHT_SHOULDER]))
        if width > 1e-6:
            return width
    for shoulder, hip in ((LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)):
        if scores[shoulder] >= min_conf and scores[hip] >= min_conf:
            length = float(np.linalg.norm(keypoints[shoulder] - keypoints[hip]))
            if length > 1e-6:
                return length * 0.75
    return None


def body_origin(keypoints: np.ndarray, scores: np.ndarray,
                min_conf: float = KEYPOINT_CONF) -> Optional[np.ndarray]:
    """Anchor point for aligning two skeletons: shoulder midpoint, else the nose."""
    if scores[LEFT_SHOULDER] >= min_conf and scores[RIGHT_SHOULDER] >= min_conf:
        return (keypoints[LEFT_SHOULDER] + keypoints[RIGHT_SHOULDER]) / 2.0
    if scores[NOSE] >= min_conf:
        return np.asarray(keypoints[NOSE], dtype=float).copy()
    return None

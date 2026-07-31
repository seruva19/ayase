"""Shared InsightFace detection helpers used by the face-identity modules.

Face-identity datasets frequently hold pre-cropped, tightly framed faces (for
example 112x112 aligned chips), which RetinaFace misses outright because there is
no context around the head. Every module that detects a face therefore retries
once on a replicate-padded copy; the detection result then refers to that padded
image, so it is returned together with the face.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_PAD_RETRY = 0.25


def _largest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def detect_largest_face(
    face_app: Any, image_bgr: np.ndarray, pad_retry: float = DEFAULT_PAD_RETRY
) -> Tuple[Optional[Any], np.ndarray]:
    """Return ``(largest face, the image its bbox/keypoints refer to)``.

    ``pad_retry`` is the replicate border added (as a fraction of the shorter
    side) for the second detection attempt; ``0`` disables the retry. The face is
    ``None`` when nothing is detected either way.
    """
    faces = face_app.get(image_bgr)
    if faces:
        return _largest(faces), image_bgr

    border = int(min(image_bgr.shape[:2]) * max(0.0, pad_retry))
    if border <= 0:
        return None, image_bgr

    padded = cv2.copyMakeBorder(
        image_bgr, border, border, border, border, cv2.BORDER_REPLICATE
    )
    faces = face_app.get(padded)
    if not faces:
        return None, image_bgr
    return _largest(faces), padded


# -- Face tracks and per-frame identity series --------------------------------
#
# The identity modules (``identity_loss``, ``dino_face_identity``) reduce a clip
# to one number computed over a handful of sampled frames, and each of them picks
# a face per frame on its own. Two questions cannot be answered from that scalar:
# WHEN identity drifted inside the clip, and WHICH person a value belongs to when
# several people share the frame. Both are ordinary questions for any
# identity-preserving generation task, so the raw material is exposed here.

#: Below this cosine similarity a detection is not considered the same person as
#: the track it is compared against.
TRACK_COS_MIN = 0.35
#: Below this box overlap a detection is not considered a geometric continuation.
TRACK_IOU_MIN = 0.3


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in box_b[:4])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


def face_tracks(
    face_app: Any,
    frames: Iterable[Tuple[int, np.ndarray]],
    *,
    iou_min: float = TRACK_IOU_MIN,
    cos_min: float = TRACK_COS_MIN,
    min_length: int = 3,
) -> List[Dict[str, Any]]:
    """Follow every face through ``frames``; return one entry per track.

    ``frames`` yields ``(frame_index, bgr_image)`` pairs; the caller decides the
    sampling stride, so a track records the indices it was actually seen on.

    Matching uses the bounding box AND the embedding together. Geometry alone
    merges two different people into one track whenever they cross or overlap --
    exactly the situation a multi-person identity check exists to catch --
    while embedding alone re-links a face after an arbitrarily long absence.

    Each track holds ``frames`` (indices), ``boxes``, ``embeddings`` and a running
    ``embedding``. Tracks shorter than ``min_length`` are dropped as detector noise.
    """
    tracks: List[Dict[str, Any]] = []
    for index, frame_bgr in frames:
        detections = face_app.get(frame_bgr) or []
        used = set()
        for track in tracks:
            if track["closed"]:
                continue
            best, best_score = -1, 0.0
            for position, detection in enumerate(detections):
                if position in used:
                    continue
                geometry = _iou(track["box"], detection.bbox)
                similarity = float(np.dot(track["embedding"], detection.normed_embedding))
                if geometry < iou_min and similarity < cos_min:
                    continue
                score = 0.5 * geometry + 0.5 * max(0.0, similarity)
                if score > best_score:
                    best, best_score = position, score
            if best < 0:
                track["closed"] = True
                continue
            detection = detections[best]
            used.add(best)
            embedding = np.asarray(detection.normed_embedding, dtype=float)
            track["box"] = tuple(float(v) for v in detection.bbox[:4])
            # Running mean, weighted towards history: a single blurred or
            # half-occluded frame should not redefine who the track is.
            merged = 0.8 * track["embedding"] + 0.2 * embedding
            track["embedding"] = merged / (np.linalg.norm(merged) + 1e-9)
            track["embeddings"].append(embedding)
            track["boxes"].append(tuple(float(v) for v in detection.bbox[:4]))
            track["frames"].append(index)
        for position, detection in enumerate(detections):
            if position in used:
                continue
            embedding = np.asarray(detection.normed_embedding, dtype=float)
            box = tuple(float(v) for v in detection.bbox[:4])
            tracks.append(
                {
                    "box": box,
                    "boxes": [box],
                    "embedding": embedding,
                    "embeddings": [embedding],
                    "frames": [index],
                    "closed": False,
                }
            )
    return [t for t in tracks if len(t["frames"]) >= min_length]


def identity_series(
    face_app: Any,
    frames: Iterable[Tuple[int, np.ndarray]],
    target_embedding: np.ndarray,
) -> List[Tuple[int, float]]:
    """Per-frame cosine similarity to ``target_embedding``: ``(frame index, value)``.

    On every frame the detection closest to the target is taken, not the largest
    face. With one person on screen the two coincide; with several they do not,
    and picking by size answers a different question than "is the target person
    still here". Frames with no detection are absent from the series rather than
    filled in, so the caller can tell a low similarity from a missing face.
    """
    target = np.asarray(target_embedding, dtype=float)
    target = target / (np.linalg.norm(target) + 1e-9)
    series: List[Tuple[int, float]] = []
    for index, frame_bgr in frames:
        detections = face_app.get(frame_bgr) or []
        if not detections:
            continue
        best = max(float(np.dot(target, d.normed_embedding)) for d in detections)
        series.append((index, best))
    return series

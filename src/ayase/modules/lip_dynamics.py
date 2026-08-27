"""Lip-motion diversity using THEval's mouth-shape distance variation.

For every detected face, the metric forms all pairwise distances among the 40
MediaPipe lip landmarks in 256x256 coordinates. It reports the mean temporal
standard deviation across those distance dimensions; higher is more dynamic.
"""

import logging
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

from ._blendshape_utils import (
    MODEL_FILENAME,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_URL,
    BlendshapeExtractor,
    select_face_index,
)

logger = logging.getLogger(__name__)

FACEMESH_LIPS = frozenset(
    {
        (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
        (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
        (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
        (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
        (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
        (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
        (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
        (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
    }
)
LIP_INDICES = tuple(sorted({index for edge in FACEMESH_LIPS for index in edge}))


def lip_shape_vector(landmarks: Sequence[Any], canvas_size: float = 256.0) -> Optional[np.ndarray]:
    """Return THEval's vector of every pairwise lip-landmark distance."""
    if len(landmarks) <= max(LIP_INDICES):
        return None
    points = np.asarray(
        [[float(landmarks[index].x), float(landmarks[index].y)] for index in LIP_INDICES],
        dtype=np.float64,
    ) * float(canvas_size)
    if not np.isfinite(points).all():
        return None
    return np.asarray(
        [np.linalg.norm(points[i] - points[j]) for i, j in combinations(range(len(points)), 2)],
        dtype=np.float64,
    )


def lip_dynamics(shape_vectors: Sequence[np.ndarray]) -> Optional[float]:
    """Average per-distance population standard deviation over time."""
    if not shape_vectors:
        return None
    vectors = np.asarray(shape_vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        return None
    score = float(np.mean(np.std(vectors, axis=0)))
    return score if math.isfinite(score) else None


class LipDynamicsModule(PipelineModule):
    """Measure talking-head mouth-shape diversity across a video."""

    name = "lip_dynamics"
    description = "THEval temporal variation of all pairwise lip-landmark distances"
    default_config = {
        "num_faces": 1,
        "face_index": None,
        "min_face_detection_confidence": 0.5,
        "min_face_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    }
    models = [
        {
            "id": MODEL_REPO_ID,
            "type": "huggingface",
            "url": MODEL_URL,
            "revision": MODEL_REVISION,
            "task": f"MediaPipe face landmarks ({MODEL_FILENAME})",
        }
    ]
    metric_info = {
        "lip_dynamics_score": (
            "THEval mean temporal standard deviation of pairwise lip-landmark distances "
            "(higher=more dynamic)"
        )
    }
    metric_groups = {"lip_dynamics_score": "face"}

    def __init__(self, config=None):
        super().__init__(config)
        self._extractor = BlendshapeExtractor(
            self.config.get("models_dir", "models"),
            num_faces=int(self.config.get("num_faces", 1)),
            face_index=self.config.get("face_index"),
            min_face_detection_confidence=float(
                self.config.get("min_face_detection_confidence", 0.5)
            ),
            min_face_presence_confidence=float(
                self.config.get("min_face_presence_confidence", 0.5)
            ),
            min_tracking_confidence=float(self.config.get("min_tracking_confidence", 0.5)),
        )
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        self._ml_available = self._extractor.setup("LipDynamics")
        self._backend = self._extractor.backend

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        try:
            score = self._score_video(sample.path)
            if score is None:
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.lip_dynamics_score = score
        except Exception as exc:
            logger.warning("LipDynamics failed for %s: %s", sample.path, exc)
        return sample

    def _score_video(self, video_path: Path) -> Optional[float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            cap.release()
            return None
        vectors = []
        previous_ms = -1
        frame_index = 0
        try:
            with self._extractor.create_landmarker() as landmarker:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    timestamp_ms = max(int(round(frame_index / fps * 1000.0)), previous_ms + 1)
                    previous_ms = timestamp_ms
                    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image = self._extractor.mediapipe.Image(
                        image_format=self._extractor.mediapipe.ImageFormat.SRGB,
                        data=rgb,
                    )
                    detected = landmarker.detect_for_video(image, timestamp_ms)
                    selected = select_face_index(detected, self._extractor.face_index)
                    if selected is not None:
                        vector = lip_shape_vector(detected.face_landmarks[selected])
                        if vector is not None:
                            vectors.append(vector)
                    frame_index += 1
        finally:
            cap.release()
        return lip_dynamics(vectors)

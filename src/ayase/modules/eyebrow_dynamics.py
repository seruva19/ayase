"""Eyebrow micro-expression intensity using the THEval landmark protocol.

The score averages frame-to-frame absolute changes in each brow-to-eye-center
distance after normalization by inter-eye distance. Higher is more dynamic.
"""

import logging
import math
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

from ._blendshape_utils import (
    MODEL_FILENAME, MODEL_REPO_ID, MODEL_REVISION, MODEL_URL,
    BlendshapeExtractor, select_face_index,
)

logger = logging.getLogger(__name__)

LEFT_BROW = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)
LEFT_EYE = (263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398)
RIGHT_BROW = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)
RIGHT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173)


def _center(landmarks: Sequence[Any], indices: Sequence[int]) -> np.ndarray:
    return np.mean(
        np.asarray([[float(landmarks[i].x), float(landmarks[i].y)] for i in indices]),
        axis=0,
    )


def normalized_brow_distances(landmarks: Sequence[Any]) -> Optional[Tuple[float, float]]:
    """Return left/right brow-to-eye distances normalized by inter-eye scale."""
    if len(landmarks) <= max(*LEFT_BROW, *LEFT_EYE, *RIGHT_BROW, *RIGHT_EYE):
        return None
    left_eye = _center(landmarks, LEFT_EYE)
    right_eye = _center(landmarks, RIGHT_EYE)
    scale = float(np.linalg.norm(left_eye - right_eye))
    if not math.isfinite(scale) or scale < 1e-6:
        return None
    left = float(np.linalg.norm(_center(landmarks, LEFT_BROW) - left_eye) / scale)
    right = float(np.linalg.norm(_center(landmarks, RIGHT_BROW) - right_eye) / scale)
    if not math.isfinite(left) or not math.isfinite(right):
        return None
    return left, right


def eyebrow_dynamics(distances: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Compute THEval's mean absolute consecutive change over both brows."""
    if len(distances) < 2:
        return None
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        return None
    return float(np.mean(np.abs(np.diff(values, axis=0))))


class EyebrowDynamicsModule(PipelineModule):
    name = "eyebrow_dynamics"
    description = "THEval inter-eye-normalized eyebrow micro-expression intensity"
    default_config = {"num_faces": 1, "face_index": None}
    models = [{
        "id": MODEL_REPO_ID, "type": "huggingface", "url": MODEL_URL,
        "revision": MODEL_REVISION, "task": f"MediaPipe face landmarks ({MODEL_FILENAME})",
    }]
    metric_info = {"eyebrow_dynamics_score": "THEval normalized eyebrow motion intensity (higher=more dynamic)"}
    metric_groups = {"eyebrow_dynamics_score": "face"}

    def __init__(self, config=None):
        super().__init__(config)
        self._extractor = BlendshapeExtractor(
            self.config.get("models_dir", "models"),
            num_faces=int(self.config.get("num_faces", 1)),
            face_index=self.config.get("face_index"),
        )
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        self._ml_available = self._extractor.setup("EyebrowDynamics")
        self._backend = self._extractor.backend

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        try:
            score = self._score_video(sample.path)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.eyebrow_dynamics_score = score
        except Exception as exc:
            logger.warning("EyebrowDynamics failed for %s: %s", sample.path, exc)
        return sample

    def _score_video(self, video_path: Path) -> Optional[float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            cap.release()
            return None
        values = []
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
                        image_format=self._extractor.mediapipe.ImageFormat.SRGB, data=rgb
                    )
                    result = landmarker.detect_for_video(image, timestamp_ms)
                    selected = select_face_index(result, self._extractor.face_index)
                    if selected is not None:
                        distances = normalized_brow_distances(result.face_landmarks[selected])
                        if distances is not None:
                            values.append(distances)
                    frame_index += 1
        finally:
            cap.release()
        return eyebrow_dynamics(values)

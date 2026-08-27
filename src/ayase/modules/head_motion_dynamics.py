"""Talking-head motion complexity using the THEval dynamics equation.

Pitch, yaw, roll, and face-center trajectories come from MediaPipe's pinned
Face Landmarker. THEval's published pose/derivative/translation aggregation is
then applied unchanged. Higher values indicate more complex head motion.
"""

import logging
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

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


def head_motion_dynamics(
    pitch: Sequence[float],
    yaw: Sequence[float],
    roll: Sequence[float],
    trans_x: Sequence[float],
    trans_y: Sequence[float],
) -> Optional[float]:
    """Apply THEval's head-motion complexity equation exactly."""
    axes = [np.asarray(values, dtype=np.float64) for values in (pitch, yaw, roll)]
    translations = [np.asarray(values, dtype=np.float64) for values in (trans_x, trans_y)]
    if not axes[0].size or any(values.size != axes[0].size for values in axes + translations):
        return None
    if not all(np.isfinite(values).all() for values in axes + translations):
        return None
    average_std = float(np.mean([np.std(values) for values in axes]))
    derivative_variances = [
        float(np.var(np.diff(values))) if values.size > 2 else 0.0 for values in axes
    ]
    average_derivative_variance = float(np.mean(derivative_variances))
    average_translation_variance = (
        float(np.mean([np.var(values) for values in translations]))
        if axes[0].size > 1
        else 0.0
    )
    score = math.sqrt(
        max(0.0, average_std * average_derivative_variance + average_translation_variance)
    )
    return score if math.isfinite(score) else None


def pose_and_center(
    matrix: np.ndarray, landmarks: Sequence[object], width: int, height: int
) -> Optional[Tuple[float, float, float, float, float]]:
    """Convert a MediaPipe facial transform and landmarks to THEval inputs."""
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not landmarks:
        return None
    angles = cv2.RQDecomp3x3(transform[:3, :3])[0]
    xs = np.asarray([float(getattr(point, "x")) for point in landmarks]) * width
    ys = np.asarray([float(getattr(point, "y")) for point in landmarks]) * height
    values = (float(angles[0]), float(angles[1]), float(angles[2]), float(np.mean(xs)), float(np.mean(ys)))
    return values if all(math.isfinite(value) for value in values) else None


class HeadMotionDynamicsModule(PipelineModule):
    name = "head_motion_dynamics"
    description = "THEval pose/derivative/translation head-motion complexity"
    default_config = {"num_faces": 1, "face_index": None}
    models = [{
        "id": MODEL_REPO_ID,
        "type": "huggingface",
        "url": MODEL_URL,
        "revision": MODEL_REVISION,
        "task": f"Face pose and landmarks ({MODEL_FILENAME})",
    }]
    metric_info = {"head_motion_dynamics_score": "THEval head pose and translation complexity (higher=more dynamic)"}
    metric_groups = {"head_motion_dynamics_score": "motion"}

    def __init__(self, config=None):
        super().__init__(config)
        self._extractor = BlendshapeExtractor(
            self.config.get("models_dir", "models"),
            num_faces=int(self.config.get("num_faces", 1)),
            face_index=self.config.get("face_index"),
            output_facial_transformation_matrixes=True,
        )
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        self._ml_available = self._extractor.setup("HeadMotionDynamics")
        self._backend = self._extractor.backend

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        try:
            score = self._score_video(sample.path)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.head_motion_dynamics_score = score
        except Exception as exc:
            logger.warning("HeadMotionDynamics failed for %s: %s", sample.path, exc)
        return sample

    def _score_video(self, video_path: Path) -> Optional[float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            cap.release()
            return None
        trajectories = [[] for _ in range(5)]
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
                    matrices = result.facial_transformation_matrixes or []
                    if selected is not None and selected < len(matrices):
                        values = pose_and_center(
                            matrices[selected], result.face_landmarks[selected], frame.shape[1], frame.shape[0]
                        )
                        if values is not None:
                            for trajectory, value in zip(trajectories, values):
                                trajectory.append(value)
                    frame_index += 1
        finally:
            cap.release()
        return head_motion_dynamics(*trajectories)

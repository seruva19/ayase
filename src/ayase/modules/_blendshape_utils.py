"""Shared MediaPipe blendshape extraction for the facial-expression metrics.

Both expression metrics read the same 52 ARKit-style coefficients from the same
pinned MediaPipe bundle, and they must keep reading them identically: a score
computed against a differently-decoded trajectory is not comparable to one that
is not, and the divergence would be silent. Extraction therefore lives here and
nowhere else. What differs between the metrics is what they do with the
trajectory afterwards, not how they obtain it.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

import cv2
import numpy as np

from ayase.config import download_model_file

logger = logging.getLogger(__name__)

MODEL_REPO_ID = "AkaneTendo25/ayase-runtime-assets"
MODEL_FILENAME = "expression_following/face_landmarker.task"
MODEL_REVISION = "409c832ac7a30524a48ab642455bf963c2a95d1f"
MODEL_URL = (
    "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/"
    f"{MODEL_REVISION}/{MODEL_FILENAME}"
)
BLENDSHAPE_DIM = 52
FPS_TOLERANCE = 1e-4
TIME_TOLERANCE = 1e-6

# Exact category names emitted by the pinned Google MediaPipe bundle. The
# neutral coefficient is part of MediaPipe's documented 52-value output.
CANONICAL_BLENDSHAPES = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)

# Coefficients driven by where the eyes point rather than by what the face is
# doing. They are part of the 52 and stay in the default feature set, but a
# caller comparing two unrelated takes may exclude them: gaze follows the shot,
# not the manner of the person in it.
GAZE_BLENDSHAPES = tuple(name for name in CANONICAL_BLENDSHAPES if name.startswith("eyeLook"))


@dataclass
class BlendshapeTrajectory:
    """Per-frame blendshape coefficients of one video, with validity per frame."""

    timestamps_sec: np.ndarray
    coefficients: np.ndarray
    valid: np.ndarray
    frame_indices: np.ndarray
    fps: float
    decoded_frames: int
    face_frames: int
    multiple_faces: bool = False

    @property
    def duration_sec(self) -> float:
        if self.timestamps_sec.size < 2:
            return 0.0
        return float(self.timestamps_sec[-1] - self.timestamps_sec[0])

    def valid_coefficients(self) -> np.ndarray:
        """Frames where a face was found, in decode order. Shape (n, 52)."""
        if self.coefficients.size == 0:
            return np.empty((0, BLENDSHAPE_DIM), dtype=np.float32)
        return cast(np.ndarray, self.coefficients[self.valid])


def categories_to_vector(categories: Sequence[Any]) -> Optional[np.ndarray]:
    """Map one frame of MediaPipe categories onto the canonical 52-vector.

    Returns ``None`` when the bundle emitted an unexpected set of names, a
    duplicate, or an out-of-range score: a partially recognised frame is dropped
    rather than silently zero-filled.
    """
    values: Dict[str, float] = {}
    for category in categories:
        name = str(category.category_name)
        if name in values:
            return None
        values[name] = float(category.score)
    if set(values) != set(CANONICAL_BLENDSHAPES):
        return None
    vector = np.asarray([values[name] for name in CANONICAL_BLENDSHAPES], dtype=np.float64)
    if vector.shape != (BLENDSHAPE_DIM,) or not np.isfinite(vector).all():
        return None
    if np.any(vector < -1e-5) or np.any(vector > 1.0 + 1e-5):
        return None
    return cast(np.ndarray, np.clip(vector, 0.0, 1.0).astype(np.float32))


def select_face_index(result: Any, face_index: Optional[int] = None) -> Optional[int]:
    """Pick which detected face to read: the configured index, else the largest.

    Selection is deterministic -- largest landmark bounding-box area, lowest
    index on ties -- so two runs over the same video read the same face.
    """
    count = len(result.face_blendshapes or [])
    landmarks = result.face_landmarks or []
    if count == 0 or len(landmarks) != count:
        return None
    if face_index is not None:
        index = int(face_index)
        return index if 0 <= index < count else None
    areas = []
    for index, face in enumerate(landmarks):
        xs = np.asarray([point.x for point in face], dtype=np.float64)
        ys = np.asarray([point.y for point in face], dtype=np.float64)
        area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
        areas.append(area if math.isfinite(area) else -1.0)
    return max(range(count), key=lambda index: (areas[index], -index))


class BlendshapeExtractor:
    """Resolves the pinned Face Landmarker bundle and decodes videos with it."""

    def __init__(
        self,
        models_dir: str = "models",
        *,
        num_faces: int = 5,
        face_index: Optional[int] = None,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_facial_transformation_matrixes: bool = False,
    ) -> None:
        self.models_dir = str(models_dir)
        self.num_faces = max(1, int(num_faces))
        self.face_index = face_index
        self.min_face_detection_confidence = float(min_face_detection_confidence)
        self.min_face_presence_confidence = float(min_face_presence_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.output_facial_transformation_matrixes = bool(
            output_facial_transformation_matrixes
        )
        self.available = False
        self.backend = "unavailable"
        self.model_path: Optional[Path] = None
        self._mp: Any = None

    def setup(self, log_prefix: str = "Blendshapes") -> bool:
        """Resolve mediapipe and the model file. Never raises; returns availability."""
        if self.available:
            return True
        self.available = False
        self.backend = "unavailable"
        try:
            import mediapipe as mp

            required = ("FaceLandmarker", "FaceLandmarkerOptions", "RunningMode")
            if not all(hasattr(mp.tasks.vision, name) for name in required):
                raise RuntimeError("installed mediapipe lacks the Face Landmarker Tasks API")
            path = download_model_file(MODEL_FILENAME, MODEL_URL, self.models_dir)
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"invalid Face Landmarker artifact: {path}")
            self.model_path = path
            self._mp = mp
            self.available = True
            self.backend = "mediapipe_face_landmarker"
            logger.info("%s: Face Landmarker bundle resolved", log_prefix)
        except ImportError:
            logger.warning("%s requires mediapipe", log_prefix)
        except Exception as e:  # noqa: BLE001 - availability probe must not raise
            self.model_path = None
            self._mp = None
            logger.warning("%s setup failed: %s", log_prefix, e)
        return self.available

    @property
    def mediapipe(self) -> Any:
        return self._mp

    def create_landmarker(self) -> Any:
        if not self.available or self._mp is None or self.model_path is None:
            raise RuntimeError("blendshape extractor is not initialized")
        options = self._mp.tasks.vision.FaceLandmarkerOptions(
            base_options=self._mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=self._mp.tasks.vision.RunningMode.VIDEO,
            num_faces=self.num_faces,
            min_face_detection_confidence=self.min_face_detection_confidence,
            min_face_presence_confidence=self.min_face_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=self.output_facial_transformation_matrixes,
        )
        return self._mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def extract(self, video_path: Path) -> BlendshapeTrajectory:
        """Decode one video into a blendshape trajectory."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            cap.release()
            raise ValueError(f"invalid_video_fps: {video_path}")

        timestamps: List[float] = []
        coefficients: List[np.ndarray] = []
        valid: List[bool] = []
        indices: List[int] = []
        multiple_faces = False
        previous_ms = -1
        frame_index = 0
        try:
            # A fresh VIDEO-mode instance per decoded video prevents timestamp
            # state from leaking when the second video starts again at t=0.
            with self.create_landmarker() as landmarker:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    time_sec = frame_index / fps
                    timestamp_ms = max(int(round(time_sec * 1000.0)), previous_ms + 1)
                    previous_ms = timestamp_ms
                    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
                    detected = landmarker.detect_for_video(image, timestamp_ms)
                    face_count = len(detected.face_blendshapes or [])
                    multiple_faces = multiple_faces or face_count > 1
                    vector = None
                    if face_count:
                        selected = select_face_index(detected, self.face_index)
                        if selected is not None:
                            vector = categories_to_vector(detected.face_blendshapes[selected])
                    timestamps.append(float(time_sec))
                    indices.append(frame_index)
                    valid.append(vector is not None)
                    coefficients.append(
                        vector if vector is not None else np.full(BLENDSHAPE_DIM, np.nan)
                    )
                    frame_index += 1
        finally:
            cap.release()

        if frame_index == 0:
            raise ValueError(f"video_decode_failure: {video_path}")
        valid_array = np.asarray(valid, dtype=bool)
        return BlendshapeTrajectory(
            timestamps_sec=np.asarray(timestamps, dtype=np.float64),
            coefficients=np.asarray(coefficients, dtype=np.float32),
            valid=valid_array,
            frame_indices=np.asarray(indices, dtype=np.int64),
            fps=fps,
            decoded_frames=frame_index,
            face_frames=int(valid_array.sum()),
            multiple_faces=multiple_faces,
        )

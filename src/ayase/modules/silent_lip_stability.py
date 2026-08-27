"""Silent-mouth stability using THEval's VAD-conditioned lip-opening MAD.

Silero VAD identifies silence runs of at least 300 ms. MediaPipe landmarks then
measure eye-distance-normalized lip opening in those frames. The reported value
is the median absolute deviation from the mean opening; lower is better.
"""

import logging
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Set

import cv2
import numpy as np

from ayase.audio import load_audio
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

UPPER_LIP_INDICES = (191, 80, 81, 82, 13, 312, 311, 310)
LOWER_LIP_INDICES = (95, 88, 178, 87, 14, 317, 402, 318)
LEFT_EYE_INDEX = 33
RIGHT_EYE_INDEX = 263


def silence_frame_indices(
    speech_segments: Iterable[Mapping[str, float]],
    fps: float,
    total_frames: int,
    minimum_silence_ms: float = 300.0,
) -> Set[int]:
    """Return frames belonging to contiguous silence runs long enough to score."""
    if not math.isfinite(fps) or fps <= 0 or total_frames <= 0:
        return set()
    silent = np.ones(total_frames, dtype=bool)
    for segment in speech_segments:
        start = max(0, int(float(segment["start"]) * fps))
        end = min(total_frames - 1, int(float(segment["end"]) * fps))
        if end >= start:
            silent[start : end + 1] = False

    minimum_frames = max(1, int(math.ceil(minimum_silence_ms / 1000.0 * fps)))
    selected: Set[int] = set()
    run_start: Optional[int] = None
    for index in range(total_frames + 1):
        is_silent = index < total_frames and bool(silent[index])
        if is_silent and run_start is None:
            run_start = index
        elif not is_silent and run_start is not None:
            if index - run_start >= minimum_frames:
                selected.update(range(run_start, index))
            run_start = None
    return selected


def normalized_lip_opening(landmarks: Sequence[Any]) -> Optional[float]:
    """Compute THEval's mean vertical lip separation normalized by eye distance."""
    if len(landmarks) <= max(RIGHT_EYE_INDEX, *UPPER_LIP_INDICES, *LOWER_LIP_INDICES):
        return None
    left = np.asarray(
        [landmarks[LEFT_EYE_INDEX].x, landmarks[LEFT_EYE_INDEX].y], dtype=np.float64
    )
    right = np.asarray(
        [landmarks[RIGHT_EYE_INDEX].x, landmarks[RIGHT_EYE_INDEX].y], dtype=np.float64
    )
    eye_distance = float(np.linalg.norm(left - right))
    if not math.isfinite(eye_distance) or eye_distance < 1e-6:
        return None
    distances = [
        abs(float(landmarks[upper].y) - float(landmarks[lower].y)) / eye_distance
        for upper, lower in zip(UPPER_LIP_INDICES, LOWER_LIP_INDICES)
    ]
    opening = float(np.mean(distances))
    return opening if math.isfinite(opening) else None


def silent_lip_mad(openings: Sequence[float]) -> Optional[float]:
    """Aggregate valid silent-frame openings exactly as THEval does."""
    values = np.asarray(openings, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    mean_opening = float(np.mean(values))
    return float(statistics.median(abs(float(value) - mean_opening) for value in values))


class SilentLipStabilityModule(PipelineModule):
    """Measure involuntary lip motion during acoustically silent intervals."""

    name = "silent_lip_stability"
    description = "THEval silent-mouth lip-opening MAD during Silero-VAD silence"
    default_config = {
        "minimum_silence_ms": 300.0,
        "sample_rate": 16000,
        "num_faces": 1,
        "face_index": None,
        "min_face_detection_confidence": 0.5,
        "min_face_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    }
    models = [
        {
            "id": "silero-vad",
            "type": "pip_package",
            "install": "pip install silero-vad",
            "task": "Speech activity detection for silence selection",
        },
        {
            "id": MODEL_REPO_ID,
            "type": "huggingface",
            "url": MODEL_URL,
            "task": "MediaPipe face landmarks",
            "revision": MODEL_REVISION,
        },
    ]
    metric_info = {
        "silent_lip_stability": (
            "THEval median absolute deviation of normalized lip opening during silence "
            "(lower=better)"
        )
    }
    metric_groups = {"silent_lip_stability": "audio"}

    def __init__(self, config=None):
        super().__init__(config)
        self.minimum_silence_ms = float(self.config.get("minimum_silence_ms", 300.0))
        self.sample_rate = int(self.config.get("sample_rate", 16000))
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
        self._vad_model: Any = None
        self._get_speech_timestamps: Any = None
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        self._ml_available = False
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad

            if not self._extractor.setup("SilentLipStability"):
                return
            self._vad_model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
            self._ml_available = True
            self._backend = "silero_vad+mediapipe_face_landmarker"
        except ImportError:
            logger.warning("SilentLipStability requires silero-vad and mediapipe")
        except Exception as exc:
            logger.warning("SilentLipStability setup failed: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        try:
            score = self._score_video(sample.path)
            if score is None:
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.silent_lip_stability = score
        except Exception as exc:
            logger.warning("SilentLipStability failed for %s: %s", sample.path, exc)
        return sample

    def _score_video(self, video_path: Path) -> Optional[float]:
        audio = load_audio(video_path, target_sr=self.sample_rate, mono=True)
        if audio is None or audio.size == 0:
            return None
        import torch

        waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        segments = self._get_speech_timestamps(
            waveform,
            self._vad_model,
            sampling_rate=self.sample_rate,
            return_seconds=True,
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        targets = silence_frame_indices(
            segments, fps, total_frames, minimum_silence_ms=self.minimum_silence_ms
        )
        if not targets:
            cap.release()
            return None

        openings = []
        previous_ms = -1
        frame_index = 0
        try:
            with self._extractor.create_landmarker() as landmarker:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_index in targets:
                        timestamp_ms = max(
                            int(round(frame_index / fps * 1000.0)), previous_ms + 1
                        )
                        previous_ms = timestamp_ms
                        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        image = self._extractor.mediapipe.Image(
                            image_format=self._extractor.mediapipe.ImageFormat.SRGB,
                            data=rgb,
                        )
                        detected = landmarker.detect_for_video(image, timestamp_ms)
                        selected = select_face_index(detected, self._extractor.face_index)
                        if selected is not None:
                            opening = normalized_lip_opening(detected.face_landmarks[selected])
                            if opening is not None:
                                openings.append(opening)
                    frame_index += 1
        finally:
            cap.release()
        return silent_lip_mad(openings)

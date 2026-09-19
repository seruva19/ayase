"""Frame-aligned facial-motion preservation after face transformation.

This module adapts the evaluation protocol published with FaceMotionPreserve
(Scientific Reports 14, 17275, 2024; DOI 10.1038/s41598-024-67989-5).  It is
intended for a generated/transformed face video and its *frame-corresponding*
source video.  It is not an identity metric and is not valid for unrelated
takes, time-shifted clips, or clips with different actions.

The paper tracks 51 non-jaw facial landmarks, forms all 1,275 landmark pairs,
and correlates their horizontal and vertical displacement trajectories. It
also reports canonical correlation and evaluates overlapping blink intervals
from eye-aspect ratio (EAR). Ayase additionally reports EAR correlation and
blink F1 as derived diagnostics. Higher is better for all correlations and
blink precision/recall/F1. Blink
scoring is disabled until ``blink_ear_threshold`` is explicitly calibrated and
configured: the paper's 0.243 operating point used different landmark geometry
and is not established for MediaPipe.

Ayase uses 51 corresponding non-jaw points from its pinned MediaPipe Face
Landmarker instead of the paper's SBR detector, whose exact landmark indices
and evaluation code were not released.  Consequently this is a clean-room
protocol adaptation, not a numerically interchangeable reproduction of the
paper.  Compare only synchronized videos with similar framing; results were
published for face de-identification of Parkinson's-disease examination videos
and were not validated by the authors as a general person-manner identifier.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
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

# Ayase's explicit MediaPipe approximation of conventional 68-point landmarks
# 17..67 (brows, nose, eyes, outer and inner lips). The paper did not publish
# its SBR landmark indices, so this mapping is not asserted to be equivalent.
LANDMARK_INDICES: Tuple[int, ...] = (
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    168, 6, 197, 195, 5, 4, 98, 97, 2, 326, 327,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
    78, 82, 13, 312, 308, 317,
)
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
PAIR_I, PAIR_J = np.triu_indices(len(LANDMARK_INDICES), k=1)
EPS = 1e-12


@dataclass
class FaceLandmarkTrajectory:
    """Per-frame 51-point positions and EAR from one decoded video."""

    positions: np.ndarray
    ear: np.ndarray
    valid: np.ndarray
    fps: float
    decoded_frames: int
    multiple_faces: bool = False


def _pearson_columns(left: np.ndarray, right: np.ndarray) -> Tuple[Optional[float], float]:
    """Mean Pearson r over columns with non-zero variance in both inputs."""
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 3:
        return None, 0.0
    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    left_norm = np.sqrt(np.sum(left_centered * left_centered, axis=0))
    right_norm = np.sqrt(np.sum(right_centered * right_centered, axis=0))
    valid = (left_norm > EPS) & (right_norm > EPS)
    if not np.any(valid):
        return None, 0.0
    values = np.sum(left_centered[:, valid] * right_centered[:, valid], axis=0)
    values /= left_norm[valid] * right_norm[valid]
    return float(np.clip(np.mean(values), -1.0, 1.0)), float(np.mean(valid))


def _first_canonical_correlation(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    """First canonical correlation of two centered 2-D trajectories."""
    if left.shape != right.shape or left.shape[0] < 3 or left.shape[1] != 2:
        return None
    x = np.asarray(left, dtype=np.float64) - np.mean(left, axis=0, keepdims=True)
    y = np.asarray(right, dtype=np.float64) - np.mean(right, axis=0, keepdims=True)
    scale = max(1, x.shape[0] - 1)
    cxx = x.T @ x / scale
    cyy = y.T @ y / scale
    cxy = x.T @ y / scale

    def inverse_sqrt(matrix: np.ndarray) -> Optional[np.ndarray]:
        values, vectors = np.linalg.eigh(matrix)
        keep = values > EPS
        if not np.any(keep):
            return None
        return (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T

    wx = inverse_sqrt(cxx)
    wy = inverse_sqrt(cyy)
    if wx is None or wy is None:
        return None
    singular = np.linalg.svd(wx @ cxy @ wy, compute_uv=False)
    if singular.size == 0 or not math.isfinite(float(singular[0])):
        return None
    return float(np.clip(singular[0], 0.0, 1.0))


def landmark_pair_correlations(
    generated: np.ndarray, reference: np.ndarray
) -> Dict[str, Optional[float]]:
    """Apply the paper's 1,275-pair dx/dy Pearson and 2-D CCA aggregation."""
    if generated.shape != reference.shape or generated.ndim != 3:
        raise ValueError("landmark trajectories must share shape (frames, 51, 2)")
    if generated.shape[1:] != (len(LANDMARK_INDICES), 2) or generated.shape[0] < 3:
        raise ValueError("expected at least three frames with 51 two-dimensional landmarks")
    generated_pairs = generated[:, PAIR_I, :] - generated[:, PAIR_J, :]
    reference_pairs = reference[:, PAIR_I, :] - reference[:, PAIR_J, :]
    x_score, x_coverage = _pearson_columns(generated_pairs[:, :, 0], reference_pairs[:, :, 0])
    y_score, y_coverage = _pearson_columns(generated_pairs[:, :, 1], reference_pairs[:, :, 1])
    generated_centered = generated_pairs - generated_pairs.mean(axis=0, keepdims=True)
    reference_centered = reference_pairs - reference_pairs.mean(axis=0, keepdims=True)
    x_valid = (
        np.linalg.norm(generated_centered[:, :, 0], axis=0) > EPS
    ) & (np.linalg.norm(reference_centered[:, :, 0], axis=0) > EPS)
    y_valid = (
        np.linalg.norm(generated_centered[:, :, 1], axis=0) > EPS
    ) & (np.linalg.norm(reference_centered[:, :, 1], axis=0) > EPS)
    cca_values: List[float] = []
    cca_valid = np.zeros(generated_pairs.shape[1], dtype=bool)
    for pair in range(generated_pairs.shape[1]):
        value = _first_canonical_correlation(
            generated_pairs[:, pair, :], reference_pairs[:, pair, :]
        )
        if value is not None:
            cca_valid[pair] = True
            cca_values.append(value)
    return {
        "face_motion_x_correlation": x_score,
        "face_motion_y_correlation": y_score,
        "face_motion_cca_correlation": (
            float(np.mean(cca_values)) if cca_values else None
        ),
        "face_motion_landmark_pair_coverage": float(np.mean(x_valid & y_valid & cca_valid)),
    }


def _eye_aspect_ratio(points: np.ndarray, indices: Sequence[int]) -> float:
    p1, p2, p3, p4, p5, p6 = (points[index] for index in indices)
    width = float(np.linalg.norm(p1 - p4))
    if width <= EPS:
        return math.nan
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * width))


def _blink_events(ear: np.ndarray, fps: float, threshold: float) -> List[Tuple[int, int]]:
    """Closed-eye runs lasting 60-700 ms, following the paper's definition."""
    values = np.asarray(ear, dtype=np.float64)
    # Missing detections break an event. Compressing them out would shorten the
    # timeline and could join two unrelated closed-eye runs into one blink.
    closed = np.isfinite(values) & (values < threshold)
    minimum = max(1, int(math.ceil(0.060 * fps)))
    maximum = max(minimum, int(math.floor(0.700 * fps)))
    events: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, is_closed in enumerate(np.append(closed, False)):
        if is_closed and start is None:
            start = index
        elif not is_closed and start is not None:
            if minimum <= index - start <= maximum:
                events.append((start, index - 1))
            start = None
    return events


def blink_overlap_scores(
    generated_ear: np.ndarray,
    reference_ear: np.ndarray,
    fps: float,
    threshold: float = 0.243,
) -> Dict[str, Any]:
    """Precision/recall/F1 when blink intervals are true positives on overlap."""
    predicted = _blink_events(generated_ear, fps, threshold)
    expected = _blink_events(reference_ear, fps, threshold)
    used = set()
    true_positive = 0
    for predicted_start, predicted_end in predicted:
        for index, (expected_start, expected_end) in enumerate(expected):
            if index not in used and max(predicted_start, expected_start) <= min(
                predicted_end, expected_end
            ):
                used.add(index)
                true_positive += 1
                break
    if expected and not predicted:
        precision, recall, f1 = 0.0, 0.0, 0.0
    elif predicted and not expected:
        precision, recall, f1 = 0.0, None, 0.0
    else:
        precision = true_positive / len(predicted) if predicted else None
        recall = true_positive / len(expected) if expected else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
    return {
        "face_motion_blink_precision": precision,
        "face_motion_blink_recall": recall,
        "face_motion_blink_f1": f1,
        "face_motion_blink_true_positives": true_positive,
        "face_motion_blink_predicted_events": len(predicted),
        "face_motion_blink_reference_events": len(expected),
    }


class FaceMotionPreservationModule(PipelineModule):
    """Compare synchronized source/transformed face motion using the published protocol."""

    name = "face_motion_preservation"
    description = "Frame-aligned FaceMotionPreserve landmark, CCA, EAR, and blink metrics"
    default_config = {
        "models_dir": "models",
        "num_faces": 2,
        "face_index": None,
        "min_paired_frames": 15,
        "low_coverage_threshold": 0.5,
        "fps_tolerance": 0.05,
        "frame_count_tolerance": 0,
        # Required to enable blink scoring. The paper reported 0.243 for its
        # SBR detector/PD cohort, but that value is not calibrated for MediaPipe.
        "blink_ear_threshold": None,
    }
    models = [{
        "id": MODEL_REPO_ID,
        "type": "huggingface",
        "url": MODEL_URL,
        "revision": MODEL_REVISION,
        "task": f"MediaPipe facial landmarks ({MODEL_FILENAME})",
        "notes": "Detector adaptation; the paper used SBR and did not publish exact indices/code",
    }]
    metric_info = {
        "face_motion_x_correlation": "Mean Pearson r of 1,275 horizontal landmark-pair trajectories (-1 to 1)",
        "face_motion_y_correlation": "Mean Pearson r of 1,275 vertical landmark-pair trajectories (-1 to 1)",
        "face_motion_cca_correlation": "Mean first canonical correlation of 2-D landmark-pair trajectories (0-1)",
        "face_motion_ear_correlation": "Ayase-derived Pearson correlation of synchronized EAR trajectories (-1 to 1)",
        "face_motion_blink_precision": "Adapted blink precision (0-1; threshold requires MediaPipe calibration)",
        "face_motion_blink_recall": "Adapted blink recall (0-1; threshold requires MediaPipe calibration)",
        "face_motion_blink_f1": "Ayase-derived blink F1 (0-1; threshold requires MediaPipe calibration)",
        "face_motion_landmark_pair_coverage": "Share of landmark pairs with defined x/y/CCA correlations (0-1)",
        "face_motion_frame_coverage": "Share of synchronized frames with a face in both videos (0-1)",
    }
    metric_groups = {field: "face" for field in metric_info}

    def __init__(self, config=None):
        super().__init__(config)
        self.min_paired_frames = max(3, int(self.config.get("min_paired_frames", 15)))
        self.low_coverage_threshold = float(self.config.get("low_coverage_threshold", 0.5))
        self.fps_tolerance = float(self.config.get("fps_tolerance", 0.05))
        self.frame_count_tolerance = max(0, int(self.config.get("frame_count_tolerance", 1)))
        threshold = self.config.get("blink_ear_threshold")
        self.blink_ear_threshold = float(threshold) if threshold is not None else None
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
        self._ml_available = self._extractor.setup("FaceMotionPreservation")
        self._backend = (
            "mediapipe_51_landmark_facemotionpreserve_adaptation"
            if self._ml_available
            else "unavailable"
        )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        if sample.reference_path is None or not Path(sample.reference_path).is_file():
            self._issue(sample, "FaceMotionPreservation requires a synchronized reference video")
            return sample
        try:
            generated = self._extract_landmarks(Path(sample.path))
            reference = self._extract_landmarks(Path(sample.reference_path))
            result = self.compare_trajectories(generated, reference)
            self._store(sample, result)
        except Exception as exc:  # noqa: BLE001 - pipeline modules must degrade gracefully
            logger.warning("FaceMotionPreservation failed for %s: %s", sample.path, exc)
            sample.metadata["face_motion_preservation_error"] = f"{type(exc).__name__}: {exc}"
            self._issue(sample, f"FaceMotionPreservation was not computed: {exc}")
        return sample

    @staticmethod
    def _issue(sample: Sample, message: str) -> None:
        sample.validation_issues.append(
            ValidationIssue(severity=ValidationSeverity.WARNING, message=message)
        )

    def _extract_landmarks(self, video_path: Path) -> FaceLandmarkTrajectory:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            cap.release()
            raise ValueError(f"invalid video fps: {video_path}")
        positions: List[np.ndarray] = []
        ears: List[float] = []
        valid: List[bool] = []
        multiple_faces = False
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
                    detected = landmarker.detect_for_video(image, timestamp_ms)
                    faces = detected.face_landmarks or []
                    multiple_faces = multiple_faces or len(faces) > 1
                    selected = select_face_index(detected, self._extractor.face_index)
                    if selected is None or selected >= len(faces):
                        positions.append(np.full((len(LANDMARK_INDICES), 2), np.nan))
                        ears.append(math.nan)
                        valid.append(False)
                    else:
                        mesh = np.asarray(
                            [(float(point.x), float(point.y)) for point in faces[selected]],
                            dtype=np.float64,
                        )
                        chosen = mesh[np.asarray(LANDMARK_INDICES)]
                        ear = float(np.mean([
                            _eye_aspect_ratio(mesh, LEFT_EYE),
                            _eye_aspect_ratio(mesh, RIGHT_EYE),
                        ]))
                        frame_valid = np.isfinite(chosen).all() and math.isfinite(ear)
                        positions.append(chosen if frame_valid else np.full_like(chosen, np.nan))
                        ears.append(ear if frame_valid else math.nan)
                        valid.append(frame_valid)
                    frame_index += 1
        finally:
            cap.release()
        if frame_index == 0:
            raise ValueError(f"video decode failure: {video_path}")
        return FaceLandmarkTrajectory(
            positions=np.asarray(positions, dtype=np.float64),
            ear=np.asarray(ears, dtype=np.float64),
            valid=np.asarray(valid, dtype=bool),
            fps=fps,
            decoded_frames=frame_index,
            multiple_faces=multiple_faces,
        )

    def compare_trajectories(
        self, generated: FaceLandmarkTrajectory, reference: FaceLandmarkTrajectory
    ) -> Dict[str, Any]:
        """Compare only frame-corresponding trajectories; never infer an alignment."""
        if abs(generated.fps - reference.fps) > self.fps_tolerance:
            raise ValueError("videos are not frame-aligned: fps differs")
        if generated.multiple_faces or reference.multiple_faces:
            raise ValueError("metric requires exactly one visible face per video")
        if abs(generated.decoded_frames - reference.decoded_frames) > self.frame_count_tolerance:
            raise ValueError("videos are not frame-aligned: frame count differs")
        frame_count = min(generated.decoded_frames, reference.decoded_frames)
        paired = generated.valid[:frame_count] & reference.valid[:frame_count]
        paired_count = int(np.sum(paired))
        frame_coverage = paired_count / frame_count if frame_count else 0.0
        if paired_count < self.min_paired_frames:
            raise ValueError("too few paired frames with a detected face")
        scores = landmark_pair_correlations(
            generated.positions[:frame_count][paired],
            reference.positions[:frame_count][paired],
        )
        ear_generated = generated.ear[:frame_count]
        ear_reference = reference.ear[:frame_count]
        ear_corr, _ = _pearson_columns(
            ear_generated[paired, None], ear_reference[paired, None]
        )
        scores["face_motion_ear_correlation"] = ear_corr
        if self.blink_ear_threshold is None or not np.all(paired):
            scores.update({
                "face_motion_blink_precision": None,
                "face_motion_blink_recall": None,
                "face_motion_blink_f1": None,
            })
        else:
            scores.update(
                blink_overlap_scores(
                    ear_generated,
                    ear_reference,
                    generated.fps,
                    self.blink_ear_threshold,
                )
            )
        scores["face_motion_frame_coverage"] = float(frame_coverage)
        scores["face_motion_paired_frames"] = paired_count
        scores["face_motion_backend"] = self._backend
        scores["face_motion_multiple_faces"] = bool(
            generated.multiple_faces or reference.multiple_faces
        )
        return scores

    def _store(self, sample: Sample, result: Dict[str, Any]) -> None:
        sample.metadata.update(result)
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        qm = sample.quality_metrics
        qm.face_motion_x_correlation = result["face_motion_x_correlation"]
        qm.face_motion_y_correlation = result["face_motion_y_correlation"]
        qm.face_motion_cca_correlation = result["face_motion_cca_correlation"]
        qm.face_motion_ear_correlation = result["face_motion_ear_correlation"]
        qm.face_motion_blink_precision = result["face_motion_blink_precision"]
        qm.face_motion_blink_recall = result["face_motion_blink_recall"]
        qm.face_motion_blink_f1 = result["face_motion_blink_f1"]
        qm.face_motion_landmark_pair_coverage = result[
            "face_motion_landmark_pair_coverage"
        ]
        qm.face_motion_frame_coverage = result["face_motion_frame_coverage"]
        for field in self.metric_info:
            if getattr(qm, field) is not None:
                qm.metric_backends[field] = self._backend
        if result["face_motion_frame_coverage"] < self.low_coverage_threshold:
            self._issue(sample, "Low paired face-landmark coverage")


__all__ = [
    "FaceLandmarkTrajectory",
    "FaceMotionPreservationModule",
    "blink_overlap_scores",
    "landmark_pair_correlations",
]

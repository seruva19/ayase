"""Facial expression following using MediaPipe blendshapes.

Compares generated and driver-video trajectories as 52 ARKit-style facial
action coefficients. Higher scores indicate more faithful expression following.
Working in blendshape space suppresses much of the identity, face-size, and
position signal that raw landmark comparison carries, but does not remove it:
distorting a face's aspect ratio while holding the expression fixed still shifts
the score, by an amount comparable to the expression range of a short clip.
Compare clips that share framing and aspect ratio, or the difference in crop is
read as a difference in expression.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

from ._blendshape_utils import (
    BLENDSHAPE_DIM,
    CANONICAL_BLENDSHAPES,
    FPS_TOLERANCE,
    MODEL_FILENAME,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_URL,
    TIME_TOLERANCE,
    BlendshapeExtractor,
    BlendshapeTrajectory,
    categories_to_vector,
    select_face_index,
)

logger = logging.getLogger(__name__)

# Decoding is shared with the other expression metric so the two cannot drift
# apart. The names below stay importable from this module, which is where
# callers reach for them.
_Trajectory = BlendshapeTrajectory

__all__ = [
    "BLENDSHAPE_DIM",
    "CANONICAL_BLENDSHAPES",
    "MODEL_FILENAME",
    "MODEL_REPO_ID",
    "MODEL_REVISION",
    "MODEL_URL",
    "ExpressionFollowing",
    "ExpressionFollowingModule",
]


class ExpressionFollowingModule(PipelineModule):
    """Measure identity-invariant facial expression/mimicry following between a
    generated video and its driver/reference video.

    In cross-driving, directly comparing normalized facial landmarks conflates
    expression fidelity with identity-specific face shape and geometry. Such a
    metric can incorrectly reward a model that leaks the driver's facial shape.
    This metric instead compares per-frame ARKit-style facial blendshape
    coefficients produced by MediaPipe Face Landmarker, then temporally aligns
    and aggregates the trajectories. Higher scores mean that the generated video
    follows the driver's expression more faithfully.

    The blendshape space suppresses face-shape signal rather than eliminating it.
    Under a controlled check — the same frames with only the face's aspect ratio
    distorted, so the expression is unchanged — the score still moved by up to
    0.05, which matched the full expression range of the clip under test. Treat
    the score as comparable only across clips with similar framing.
    """

    name = "expression_following"
    description = "Driver-expression fidelity via MediaPipe blendshapes (identity-suppressed)"
    default_config = {
        "models_dir": "models",
        "min_face_detection_confidence": 0.5,
        "min_face_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "low_coverage_threshold": 0.5,
        "num_faces": 5,
        "face_index": None,
    }
    metric_groups = {
        "expression_following": "face",
        "expression_following_distance": "face",
        "expression_following_coverage": "face",
    }
    models = [
        {
            "id": MODEL_REPO_ID,
            "type": "huggingface",
            "task": "MediaPipe Face Landmarker ARKit-style blendshapes",
            "url": MODEL_URL,
            "auto_download": "yes",
            "notes": f"Pinned revision {MODEL_REVISION}; file {MODEL_FILENAME}",
        }
    ]
    metric_info = {
        "expression_following": "Mean aligned blendshape fidelity (0-1, higher=better)",
        "expression_following_distance": "Mean normalized-L1 blendshape distance (0-1, lower=better)",
        "expression_following_coverage": "Joint valid-face coverage on the common time grid (0-1)",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.models_dir = str(self.config.get("models_dir", "models"))
        self.low_coverage_threshold = float(self.config.get("low_coverage_threshold", 0.5))
        self.num_faces = max(1, int(self.config.get("num_faces", 5)))
        self.face_index = self.config.get("face_index")
        self._extractor = BlendshapeExtractor(
            self.models_dir,
            num_faces=self.num_faces,
            face_index=self.face_index,
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

    @property
    def _model_path(self) -> Optional[Path]:
        return self._extractor.model_path

    @property
    def _mp(self) -> Any:
        return self._extractor.mediapipe

    def setup(self) -> None:
        self._ml_available = self._extractor.setup("ExpressionFollowing")
        self._backend = self._extractor.backend

    def _create_landmarker(self) -> Any:
        if not self._ml_available:
            raise RuntimeError("ExpressionFollowing is not initialized")
        return self._extractor.create_landmarker()

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        if sample.reference_path is None:
            self._input_issue(sample, "A driver/reference video is required")
            return sample
        reference = Path(sample.reference_path)
        if not reference.is_file():
            self._input_issue(sample, "Driver/reference video is missing or unreadable")
            return sample

        try:
            generation = self._extract_blendshapes(Path(sample.path))
            driver = self._extract_blendshapes(reference)
            result = self._compare_trajectories(generation, driver)
            self._store_result(sample, result)
        except Exception as e:
            logger.warning("ExpressionFollowing failed for %s: %s", sample.path, e)
            sample.metadata["expression_following_error"] = f"{type(e).__name__}: {e}"
        return sample

    @staticmethod
    def _input_issue(sample: Sample, message: str) -> None:
        sample.validation_issues.append(
            ValidationIssue(severity=ValidationSeverity.WARNING, message=message)
        )

    def _extract_blendshapes(self, video_path: Path) -> _Trajectory:
        return self._extractor.extract(video_path)

    def _select_face_index(self, result: Any) -> Optional[int]:
        return select_face_index(result, self.face_index)

    @staticmethod
    def _categories_to_vector(categories: Sequence[Any]) -> Optional[np.ndarray]:
        return categories_to_vector(categories)

    @staticmethod
    def _normalized_l1(left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != (BLENDSHAPE_DIM,) or right.shape != (BLENDSHAPE_DIM,):
            raise ValueError("blendshape vectors must have 52 coefficients")
        return float(np.clip(np.mean(np.abs(left - right)), 0.0, 1.0))

    @staticmethod
    def _resample(trajectory: _Trajectory, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        output = np.full((len(grid), BLENDSHAPE_DIM), np.nan, dtype=np.float32)
        output_valid = np.zeros(len(grid), dtype=bool)
        times = trajectory.timestamps_sec
        for target_index, target in enumerate(grid):
            exact = np.flatnonzero(np.abs(times - target) <= TIME_TOLERANCE)
            if exact.size:
                source_index = int(exact[0])
                if trajectory.valid[source_index]:
                    output[target_index] = trajectory.coefficients[source_index]
                    output_valid[target_index] = True
                continue
            right = int(np.searchsorted(times, target, side="right"))
            left = right - 1
            if left < 0 or right >= len(times):
                continue
            # Interpolate only adjacent decoded frames. This intentionally does
            # not bridge intervals containing a missing-face observation.
            if (
                not trajectory.valid[left]
                or not trajectory.valid[right]
                or trajectory.frame_indices[right] - trajectory.frame_indices[left] != 1
                or times[right] - times[left] > 1.5 / trajectory.fps + TIME_TOLERANCE
            ):
                continue
            alpha = float((target - times[left]) / (times[right] - times[left]))
            output[target_index] = (
                (1.0 - alpha) * trajectory.coefficients[left]
                + alpha * trajectory.coefficients[right]
            )
            output_valid[target_index] = True
        return output, output_valid

    def _compare_trajectories(self, generation: _Trajectory, driver: _Trajectory) -> dict:
        flags: List[str] = []
        overlap_start = max(float(generation.timestamps_sec[0]), float(driver.timestamps_sec[0]))
        overlap_end = min(float(generation.timestamps_sec[-1]), float(driver.timestamps_sec[-1]))
        if overlap_end < overlap_start:
            grid = np.asarray([], dtype=np.float64)
            flags.append("no_common_overlap")
        else:
            target_fps = min(generation.fps, driver.fps)
            count = int(math.floor((overlap_end - overlap_start) * target_fps + 1e-6)) + 1
            grid = overlap_start + np.arange(count, dtype=np.float64) / target_fps
            grid = grid[grid <= overlap_end + TIME_TOLERANCE]

        gen_values, gen_valid = self._resample(generation, grid)
        ref_values, ref_valid = self._resample(driver, grid)
        joint_valid = gen_valid & ref_valid
        per_frame: List[Optional[float]] = [None] * len(grid)
        distances: List[float] = []
        for index in np.flatnonzero(joint_valid):
            frame_distance = self._normalized_l1(gen_values[index], ref_values[index])
            distances.append(frame_distance)
            per_frame[int(index)] = float(np.clip(1.0 - frame_distance, 0.0, 1.0))

        total = int(len(grid))
        valid_count = int(joint_valid.sum())
        coverage = float(valid_count / total) if total else 0.0
        mean_distance: Optional[float] = float(np.mean(distances)) if distances else None
        score = float(1.0 - mean_distance) if mean_distance is not None else None

        if not gen_valid.any():
            flags.append("no_face_generation")
        if not ref_valid.any():
            flags.append("no_face_reference")
        if valid_count == 0:
            flags.append("no_joint_valid_frames")
        if coverage < self.low_coverage_threshold:
            flags.append("low_face_visibility")
        if abs(generation.fps - driver.fps) > FPS_TOLERANCE:
            flags.append("fps_mismatch_resampled")
        if (
            generation.decoded_frames != driver.decoded_frames
            or abs(generation.duration_sec - driver.duration_sec) > 1.0 / min(generation.fps, driver.fps)
        ):
            flags.append("length_mismatch_common_overlap")
        if generation.multiple_faces:
            flags.append("multiple_faces_generation")
        if driver.multiple_faces:
            flags.append("multiple_faces_reference")

        return {
            "expression_following": score,
            "expression_following_distance": mean_distance,
            "expression_following_per_frame": per_frame,
            "expression_following_coverage": coverage,
            "expression_following_valid_frames": valid_count,
            "expression_following_total_frames": total,
            "expression_following_timestamps_sec": [float(value) for value in grid],
            "expression_following_flags": flags,
            "expression_following_distance_type": "normalized_l1",
            "expression_following_feature_type": "mediapipe_arkit_blendshapes_52",
            "expression_following_alignment": "common_overlap_min_fps_linear_no_gap_bridge",
            "expression_following_target_fps": float(min(generation.fps, driver.fps)),
            "expression_following_generation_fps": float(generation.fps),
            "expression_following_reference_fps": float(driver.fps),
            "expression_following_generation_decoded_frames": generation.decoded_frames,
            "expression_following_reference_decoded_frames": driver.decoded_frames,
            "expression_following_generation_face_frames": generation.face_frames,
            "expression_following_reference_face_frames": driver.face_frames,
            "expression_following_overlap_duration_sec": (
                float(overlap_end - overlap_start) if total else 0.0
            ),
        }

    @staticmethod
    def _store_result(sample: Sample, result: dict) -> None:
        sample.metadata.update(result)
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.expression_following = result["expression_following"]
        sample.quality_metrics.expression_following_distance = result[
            "expression_following_distance"
        ]
        sample.quality_metrics.expression_following_coverage = result[
            "expression_following_coverage"
        ]


# Short alias; the registry-facing convention uses *Module.
ExpressionFollowing = ExpressionFollowingModule

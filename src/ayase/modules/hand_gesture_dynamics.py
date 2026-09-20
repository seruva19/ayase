"""Reference-relative 2D hand and finger dynamics without frame alignment.

Intended use: compare the distributions and dynamics of visible hand poses in
two predominantly single-person videos.  ``sample.path`` and
``sample.reference_path`` are sampled independently, so duration, frame rate,
and individual frames need not correspond.  The module accepts a sampled frame
only when the pinned RTMLib/DWPose Wholebody backend returns exactly one person;
COCO-WholeBody joints 91:112 and 112:133 are used as the left and right hands.

Each usable hand is translated to its wrist and divided by the median distance
from the wrist to the confident index, middle, ring, and little-finger MCP
joints.  Articulation is the median straight-line/base-to-tip distance divided
by polyline finger length (0 only when the tip returns to its base, 1 straight)
across usable fingers.
Openness is the median wrist-to-fingertip distance in palm scales.  Pinch is the
thumb-tip-to-index-tip distance in palm scales.  For each descriptor, location
is its median and amplitude is its 90th-minus-10th percentile span; the reported
differences are absolute sample/reference differences, where 0 means equal.
Articulation location/amplitude differences are in [0, 1]; openness and pinch
differences are in [0, +inf) palm scales.  Articulation speed is the median
per-second displacement of jointly observed, wrist-centred, palm-normalized
hand joints; its difference is in [0, +inf) palm scales per second and missing
frames break runs.  The left/right speed-asymmetry difference is in [0, 1].
Coverage is the fraction of sampled frames with at least one normalizable hand,
while joint observability is the confident finite fraction of all 42 hand
joints.  All four coverage/observability outputs are in [0, 1], higher means
more observable.

These are separate 2D landmark diagnostics, not an aggregate quality score, a
validated perceptual metric, an identity/person metric, or a measure of gesture
semantics.  Results are affected by occlusion, motion blur, tiny hands, depth
and viewpoint changes, mirroring/handedness, gloves, camera motion/rotation, and
the detector's training domain.  Palm normalization removes image translation
and uniform scale but cannot recover 3D pose or resolve left/right swaps.  The
module deliberately does not report wrist workspace, which belongs to body
motion metrics.

Primary sources: DWPose (Yang et al., 2023, arXiv:2307.15880) and the
COCO-WholeBody 133-keypoint convention (Jin et al., ECCV 2020).  The descriptors
and distribution comparisons above are explicitly local, transparent choices;
the cited works provide the landmark estimator and schema, not validation of
these diagnostics as human-similarity judgements.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.modules.pose_heat_ssim import (
    DWPose_DETECTOR,
    DWPose_DETECTOR_SHA256,
    DWPose_POSE,
    DWPose_POSE_SHA256,
    DWPose_REPO,
    DWPose_REVISION,
    single_wholebody_pose,
    verified_dwpose_asset,
)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

BODY_SLICE = slice(0, 17)
LEFT_HAND_SLICE = slice(91, 112)
RIGHT_HAND_SLICE = slice(112, 133)
HAND_JOINTS = 21
PALM_MCPS = (5, 9, 13, 17)
FINGERTIPS = (4, 8, 12, 16, 20)
FINGER_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)


@dataclass(frozen=True)
class HandMoment:
    """One sampled moment; absent hands remain explicit temporal gaps."""

    timestamp: float
    left: Optional[np.ndarray]
    left_valid: Optional[np.ndarray]
    right: Optional[np.ndarray]
    right_valid: Optional[np.ndarray]


@dataclass(frozen=True)
class HandTrack:
    """Palm-normalized hand observations and clip-level detector coverage."""

    moments: Tuple[HandMoment, ...]
    coverage: float
    joint_observability: float
    attempted: int


@dataclass(frozen=True)
class HandDescriptors:
    """Transparent scalar shape descriptors for one observed hand."""

    articulation: Optional[float]
    openness: Optional[float]
    pinch: Optional[float]


def normalize_hand(
    points: np.ndarray,
    scores: np.ndarray,
    min_conf: float,
    *,
    min_palm_points: int = 3,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Wrist-centre and palm-normalize one 21-joint COCO-WholeBody hand."""

    points = np.asarray(points, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if points.shape != (HAND_JOINTS, 2) or scores.shape != (HAND_JOINTS,):
        return None
    finite = np.isfinite(points).all(axis=1) & np.isfinite(scores)
    valid = finite & (scores >= min_conf)
    if not valid[0]:
        return None
    palm_distances = [
        float(np.linalg.norm(points[index] - points[0]))
        for index in PALM_MCPS
        if valid[index]
    ]
    palm_distances = [value for value in palm_distances if math.isfinite(value) and value > 1e-12]
    required_palm_points = min(len(PALM_MCPS), max(3, int(min_palm_points)))
    if len(palm_distances) < required_palm_points:
        return None
    scale = float(np.median(palm_distances))
    if not math.isfinite(scale) or scale <= 1e-12:
        return None
    normalized = (points - points[0]) / scale
    normalized[~valid] = np.nan
    return normalized, valid


def hand_descriptors(points: np.ndarray, valid: np.ndarray) -> HandDescriptors:
    """Compute articulation, openness, and pinch from a normalized hand."""

    points = np.asarray(points, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if points.shape != (HAND_JOINTS, 2) or valid.shape != (HAND_JOINTS,):
        return HandDescriptors(None, None, None)

    straightness: List[float] = []
    for chain in FINGER_CHAINS:
        if not all(valid[index] and np.isfinite(points[index]).all() for index in chain):
            continue
        segments = np.diff(points[np.asarray(chain)], axis=0)
        path_length = float(np.linalg.norm(segments, axis=1).sum())
        if path_length <= 1e-12:
            continue
        chord = float(np.linalg.norm(points[chain[-1]] - points[chain[0]]))
        straightness.append(float(np.clip(chord / path_length, 0.0, 1.0)))
    articulation = float(np.median(straightness)) if len(straightness) >= 3 else None

    tip_distances = [
        float(np.linalg.norm(points[index]))
        for index in FINGERTIPS
        if valid[index] and np.isfinite(points[index]).all()
    ]
    openness = float(np.median(tip_distances)) if len(tip_distances) >= 3 else None

    pinch = None
    if valid[4] and valid[8] and np.isfinite(points[[4, 8]]).all():
        pinch = float(np.linalg.norm(points[4] - points[8]))
    return HandDescriptors(articulation, openness, pinch)


def _shape_speeds(
    moments: Sequence[HandMoment],
    side: str,
    min_velocity_joints: int,
) -> np.ndarray:
    """Median normalized joint velocity for adjacent observations of one hand."""

    speeds: List[float] = []
    points_name = side
    valid_name = f"{side}_valid"
    for previous, current in zip(moments, moments[1:]):
        previous_points = getattr(previous, points_name)
        current_points = getattr(current, points_name)
        previous_valid = getattr(previous, valid_name)
        current_valid = getattr(current, valid_name)
        if (
            previous_points is None
            or current_points is None
            or previous_valid is None
            or current_valid is None
        ):
            continue
        dt = current.timestamp - previous.timestamp
        if not math.isfinite(dt) or dt <= 0.0:
            continue
        jointly_valid = np.asarray(previous_valid) & np.asarray(current_valid)
        jointly_valid[0] = False  # Wrist is identically zero after centring.
        indices = np.flatnonzero(jointly_valid)
        if len(indices) < max(1, int(min_velocity_joints)):
            continue
        displacement = np.linalg.norm(
            np.asarray(current_points)[indices] - np.asarray(previous_points)[indices], axis=1
        )
        displacement = displacement[np.isfinite(displacement)]
        if len(displacement) < max(1, int(min_velocity_joints)):
            continue
        speeds.append(float(np.median(displacement) / dt))
    return np.asarray(speeds, dtype=np.float64)


def _location_amplitude(values: Sequence[float], min_samples: int) -> Optional[Tuple[float, float]]:
    """Median location and robust p90-p10 amplitude for finite values."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) < max(1, int(min_samples)):
        return None
    low, high = np.percentile(array, (10.0, 90.0))
    return float(np.median(array)), float(max(0.0, high - low))


def summarize_hand_track(
    track: HandTrack,
    *,
    min_samples: int,
    min_speed_samples: int,
    min_velocity_joints: int,
) -> Dict[str, Optional[float]]:
    """Summarize one clip independently; no cross-clip frame pairing occurs."""

    descriptor_values: Dict[str, List[float]] = {
        "articulation": [],
        "openness": [],
        "pinch": [],
    }
    for moment in track.moments:
        for side in ("left", "right"):
            points = getattr(moment, side)
            valid = getattr(moment, f"{side}_valid")
            if points is None or valid is None:
                continue
            descriptors = hand_descriptors(points, valid)
            for name in descriptor_values:
                value = getattr(descriptors, name)
                if value is not None and math.isfinite(value):
                    descriptor_values[name].append(value)

    summary: Dict[str, Optional[float]] = {}
    for name, values in descriptor_values.items():
        result = _location_amplitude(values, min_samples)
        summary[f"{name}_location"] = result[0] if result is not None else None
        summary[f"{name}_amplitude"] = result[1] if result is not None else None

    left_speeds = _shape_speeds(track.moments, "left", min_velocity_joints)
    right_speeds = _shape_speeds(track.moments, "right", min_velocity_joints)
    pooled = np.concatenate((left_speeds, right_speeds))
    summary["speed"] = (
        float(np.median(pooled)) if len(pooled) >= max(1, int(min_speed_samples)) else None
    )
    if (
        len(left_speeds) >= max(1, int(min_speed_samples))
        and len(right_speeds) >= max(1, int(min_speed_samples))
    ):
        left = float(np.median(left_speeds))
        right = float(np.median(right_speeds))
        total = left + right
        summary["left_right_asymmetry"] = 0.0 if total <= 1e-12 else abs(left - right) / total
    else:
        summary["left_right_asymmetry"] = None
    return summary


def compare_hand_tracks(
    sample_track: HandTrack,
    reference_track: HandTrack,
    *,
    min_samples: int = 4,
    min_speed_samples: int = 3,
    min_velocity_joints: int = 8,
) -> Dict[str, Optional[float]]:
    """Compare independently summarized hand tracks with absolute differences."""

    result: Dict[str, Optional[float]] = {
        "hand_gesture_sample_coverage": float(np.clip(sample_track.coverage, 0.0, 1.0)),
        "hand_gesture_reference_coverage": float(
            np.clip(reference_track.coverage, 0.0, 1.0)
        ),
        "hand_gesture_sample_joint_observability": float(
            np.clip(sample_track.joint_observability, 0.0, 1.0)
        ),
        "hand_gesture_reference_joint_observability": float(
            np.clip(reference_track.joint_observability, 0.0, 1.0)
        ),
    }
    sample_summary = summarize_hand_track(
        sample_track,
        min_samples=min_samples,
        min_speed_samples=min_speed_samples,
        min_velocity_joints=min_velocity_joints,
    )
    reference_summary = summarize_hand_track(
        reference_track,
        min_samples=min_samples,
        min_speed_samples=min_speed_samples,
        min_velocity_joints=min_velocity_joints,
    )
    for descriptor in ("articulation", "openness", "pinch"):
        for statistic in ("location", "amplitude"):
            key = f"{descriptor}_{statistic}"
            sample_value = sample_summary[key]
            reference_value = reference_summary[key]
            field = f"hand_gesture_{descriptor}_{statistic}_difference"
            result[field] = (
                abs(sample_value - reference_value)
                if sample_value is not None and reference_value is not None
                else None
            )
    sample_speed = sample_summary["speed"]
    reference_speed = reference_summary["speed"]
    result["hand_gesture_articulation_speed_difference"] = (
        abs(sample_speed - reference_speed)
        if sample_speed is not None and reference_speed is not None
        else None
    )
    sample_asymmetry = sample_summary["left_right_asymmetry"]
    reference_asymmetry = reference_summary["left_right_asymmetry"]
    result["hand_gesture_left_right_asymmetry_difference"] = (
        abs(sample_asymmetry - reference_asymmetry)
        if sample_asymmetry is not None and reference_asymmetry is not None
        else None
    )
    return result


class HandGestureDynamicsModule(PipelineModule):
    """Compare transparent hand-shape distributions and dynamics."""

    name = "hand_gesture_dynamics"
    description = "Reference-relative 2D hand/finger distribution and dynamics diagnostics"
    default_config = {
        "device": "auto",
        "models_dir": "models",
        "moments": 64,
        "min_conf": 0.3,
        "min_palm_points": 3,
        "min_samples": 4,
        "min_speed_samples": 3,
        "min_velocity_joints": 8,
    }
    models = [
        {
            "id": DWPose_REPO,
            "type": "huggingface",
            "task": "Official DWPose detector and 133-keypoint COCO-WholeBody estimator",
            "revision": DWPose_REVISION,
            "files": [DWPose_DETECTOR, DWPose_POSE],
            "size": "351.1 MB total",
            "auto_download": True,
            "license": "Apache-2.0",
            "notes": (
                f"SHA-256 {DWPose_DETECTOR}={DWPose_DETECTOR_SHA256}; "
                f"{DWPose_POSE}={DWPose_POSE_SHA256}"
            ),
        },
        {
            "id": "rtmlib>=0.0.13",
            "type": "pip_package",
            "install": "pip install rtmlib",
            "task": "ONNX detector and Wholebody pose runtime",
        },
    ]
    metric_info = {
        "hand_gesture_sample_coverage": "Sample frames with >=1 normalizable hand (0-1)",
        "hand_gesture_reference_coverage": "Reference frames with >=1 normalizable hand (0-1)",
        "hand_gesture_sample_joint_observability": (
            "Confident finite sample hand joints among 42 per sampled frame (0-1)"
        ),
        "hand_gesture_reference_joint_observability": (
            "Confident finite reference hand joints among 42 per sampled frame (0-1)"
        ),
        "hand_gesture_articulation_location_difference": (
            "Absolute median finger-straightness difference (0=equal; range 0-1)"
        ),
        "hand_gesture_articulation_amplitude_difference": (
            "Absolute p90-p10 finger-straightness span difference (0=equal; range 0-1)"
        ),
        "hand_gesture_openness_location_difference": (
            "Absolute median palm-normalized openness difference (0=equal)"
        ),
        "hand_gesture_openness_amplitude_difference": (
            "Absolute p90-p10 palm-normalized openness span difference (0=equal)"
        ),
        "hand_gesture_pinch_location_difference": (
            "Absolute median palm-normalized thumb-index distance difference (0=equal)"
        ),
        "hand_gesture_pinch_amplitude_difference": (
            "Absolute p90-p10 palm-normalized pinch-distance span difference (0=equal)"
        ),
        "hand_gesture_articulation_speed_difference": (
            "Absolute median normalized hand-shape speed difference per second (0=equal)"
        ),
        "hand_gesture_left_right_asymmetry_difference": (
            "Absolute difference in normalized left/right shape-speed asymmetry (0=equal; range 0-1)"
        ),
    }
    metric_groups = {field: "motion" for field in metric_info}

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.moments = max(2, int(self.config.get("moments", 64)))
        self.min_conf = float(self.config.get("min_conf", 0.3))
        self.min_palm_points = max(1, int(self.config.get("min_palm_points", 3)))
        self.min_samples = max(1, int(self.config.get("min_samples", 4)))
        self.min_speed_samples = max(1, int(self.config.get("min_speed_samples", 3)))
        self.min_velocity_joints = max(
            1, int(self.config.get("min_velocity_joints", 8))
        )
        self._backend_available = False
        self._wholebody = None
        self._device = "cpu"
        self._backend_label = f"dwpose:{DWPose_REVISION}/rtmlib"

    def setup(self) -> None:
        """Check RTMLib availability; defer model construction until processing."""

        if self.test_mode:
            return
        try:
            from rtmlib import Wholebody  # noqa: F401

            requested = str(self.config.get("device", "auto"))
            if requested == "auto":
                from ayase.runtime import resolve_torch_device

                requested = str(resolve_torch_device("auto"))
            self._device = "cuda" if requested.startswith("cuda") else "cpu"
            self._backend_available = True
        except ImportError:
            logger.warning("hand_gesture_dynamics unavailable: install rtmlib")
        except Exception as exc:
            logger.warning("hand_gesture_dynamics backend check failed: %s", exc)

    def _get_backend(self):
        """Return an injected backend or shared pinned DWPose Wholebody backend."""

        if self._wholebody is not None:
            return self._wholebody
        if not self._backend_available:
            return None

        from ayase.runtime import shared_runtime_resource

        def build():
            from rtmlib import Wholebody

            models_dir = str(self.config.get("models_dir", "models"))
            detector = verified_dwpose_asset(
                DWPose_DETECTOR, DWPose_DETECTOR_SHA256, models_dir
            )
            pose = verified_dwpose_asset(DWPose_POSE, DWPose_POSE_SHA256, models_dir)
            return Wholebody(
                det=str(detector),
                pose=str(pose),
                det_input_size=(640, 640),
                pose_input_size=(288, 384),
                to_openpose=False,
                backend="onnxruntime",
                device=self._device,
            )

        self._wholebody = shared_runtime_resource(
            self,
            ("pose_heat_ssim_dwpose", DWPose_REVISION, "onnxruntime", self._device, False),
            build,
        )
        return self._wholebody

    def _extract_hand(
        self, points: np.ndarray, scores: np.ndarray, hand_slice: slice
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return normalize_hand(
            points[hand_slice],
            scores[hand_slice],
            self.min_conf,
            min_palm_points=self.min_palm_points,
        )

    def _track(self, path: Path, backend) -> HandTrack:
        """Sample one video and preserve temporal gaps and per-clip coverage."""

        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            return HandTrack((), 0.0, 0.0, 0)
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if frame_count <= 1 or not math.isfinite(fps) or fps <= 0.0:
                return HandTrack((), 0.0, 0.0, 0)
            indices = np.unique(
                np.linspace(0, frame_count - 1, min(self.moments, frame_count), dtype=int)
            )
            moments: List[HandMoment] = []
            usable_frames = 0
            observed_joints = 0
            previous_timestamp = -math.inf
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                decoder_time = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                timestamp = decoder_time
                fallback_time = float(frame_index) / fps
                if (
                    not math.isfinite(timestamp)
                    or timestamp < 0.0
                    or timestamp <= previous_timestamp
                ):
                    timestamp = fallback_time
                previous_timestamp = timestamp
                if not ok:
                    moments.append(HandMoment(timestamp, None, None, None, None))
                    continue
                try:
                    pose = single_wholebody_pose(backend(np.ascontiguousarray(frame)))
                except Exception as exc:
                    logger.debug("hand_gesture_dynamics skipped a pose failure: %s", exc)
                    moments.append(HandMoment(timestamp, None, None, None, None))
                    continue
                if pose is None:
                    moments.append(HandMoment(timestamp, None, None, None, None))
                    continue
                points, scores = pose
                hand_points = points[91:133]
                hand_scores = scores[91:133]
                observed_joints += int(
                    np.sum(
                        np.isfinite(hand_points).all(axis=1)
                        & np.isfinite(hand_scores)
                        & (hand_scores >= self.min_conf)
                    )
                )
                left = self._extract_hand(points, scores, LEFT_HAND_SLICE)
                right = self._extract_hand(points, scores, RIGHT_HAND_SLICE)
                if left is not None or right is not None:
                    usable_frames += 1
                moments.append(
                    HandMoment(
                        timestamp,
                        left[0] if left is not None else None,
                        left[1] if left is not None else None,
                        right[0] if right is not None else None,
                        right[1] if right is not None else None,
                    )
                )
            attempted = len(indices)
            return HandTrack(
                tuple(moments),
                usable_frames / attempted if attempted else 0.0,
                observed_joints / float(attempted * 2 * HAND_JOINTS) if attempted else 0.0,
                attempted,
            )
        finally:
            capture.release()

    def _compare(self, sample_path: Path, reference_path: Path, backend):
        sample_track = self._track(sample_path, backend)
        reference_track = self._track(reference_path, backend)
        if sample_track.attempted <= 0 or reference_track.attempted <= 0:
            return None
        return compare_hand_tracks(
            sample_track,
            reference_track,
            min_samples=self.min_samples,
            min_speed_samples=self.min_speed_samples,
            min_velocity_joints=self.min_velocity_joints,
        )

    def process(self, sample: Sample) -> Sample:
        """Attach available diagnostics and always return the original sample."""

        if not sample.is_video or sample.reference_path is None:
            return sample
        sample_path = Path(sample.path)
        reference_path = Path(sample.reference_path)
        if not sample_path.is_file() or not reference_path.is_file():
            return sample
        try:
            backend = self._get_backend()
            if backend is None:
                return sample
            result = self._compare(sample_path, reference_path, backend)
            if result is None:
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            metrics = sample.quality_metrics
            for field, value in result.items():
                if value is None:
                    continue
                setattr(metrics, field, round(float(value), 6))
                metrics.metric_backends[field] = self._backend_label
        except Exception as exc:
            logger.warning("hand_gesture_dynamics failed for %s: %s", sample.path, exc)
        return sample

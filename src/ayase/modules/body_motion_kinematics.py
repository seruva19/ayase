"""Reference-relative 2D body-motion kinematics without frame correspondence.

Intended use: describe whether a generated video has similar *amounts* of body
motion to a reference video.  The exact inputs are ``sample.path`` (generated
video) and ``sample.reference_path`` (reference video).  In every sampled frame,
the largest detected person is treated as the single prominent subject.  Each
clip is summarized independently, so their frames, durations, and performed
movements do not need to correspond.

The measurements apply only where a visible COCO-17 2D pose can be estimated.
They are affected by camera motion, viewpoint, framing, foreshortening, pose
estimation error, and occlusion.  They do not measure identity or action
semantics, and are not evidence of a person-specific motion style.  Multi-person
scenes are outside the intended use because choosing the largest detection does
not constitute identity tracking.

Positions are root-centered and divided by the per-frame body scale.  First,
second, and third finite differences use timestamps in seconds.  A missing pose
or keypoint ends a run; differences never bridge an unobserved sample.  For
background on finite differences on nonuniform grids, see B. Fornberg,
"Generation of Finite Difference Formulas on Arbitrarily Spaced Grids",
Mathematics of Computation 51(184), 1988, pp. 699-706.

Outputs are deliberately separate diagnostics rather than an unvalidated
aggregate.  Speed, acceleration, jerk, and range are generated/reference ratios
in ``[0, +inf)`` (1 means equal; above 1 means more in the generated clip).
Ratios are unset when only the reference statistic is zero.  Left/right symmetry
and idle-fraction differences are in ``[0, 1]`` (0 means equal).  Pose and arm
coverage are the lower coverage of the two clips, also in ``[0, 1]``.  No claims
of identity recognition, action recognition, or person-specific validation are
made for these diagnostics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_LEFT_JOINTS = (5, 7, 9, 11, 13, 15)
_RIGHT_JOINTS = (6, 8, 10, 12, 14, 16)
_ARM_CHAINS = ((5, 7, 9), (6, 8, 10))
_KINEMATIC_JOINTS = tuple(range(17))


@dataclass(frozen=True)
class _PoseMoment:
    """One requested video moment; ``points=None`` records a detection gap."""

    timestamp: float
    points: Optional[np.ndarray]
    scores: Optional[np.ndarray]


@dataclass(frozen=True)
class _ClipTrack:
    """Sampled normalized poses and coverage for one clip."""

    moments: Tuple[_PoseMoment, ...]
    pose_coverage: float
    arm_coverage: float
    attempted: int


def _differentiate(values: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """First divided differences and their midpoint timestamps."""
    dt = np.diff(times)
    valid = np.isfinite(dt) & (dt > 0.0)
    if not np.any(valid):
        return np.empty((0, values.shape[1]), dtype=np.float64), np.empty(0)
    differences = np.diff(values, axis=0)[valid] / dt[valid, None]
    midpoints = ((times[:-1] + times[1:]) * 0.5)[valid]
    finite = np.isfinite(differences).all(axis=1) & np.isfinite(midpoints)
    return differences[finite], midpoints[finite]


def _joint_derivative_magnitudes(
    moments: Sequence[_PoseMoment],
    joints: Sequence[int],
    order: int,
    min_conf: float,
) -> np.ndarray:
    """Derivative magnitudes from contiguous confident per-joint runs."""
    magnitudes: List[float] = []
    for joint in joints:
        run_points: List[np.ndarray] = []
        run_times: List[float] = []

        def finish_run() -> None:
            if len(run_points) <= order:
                return
            values = np.asarray(run_points, dtype=np.float64)
            times = np.asarray(run_times, dtype=np.float64)
            for _ in range(order):
                values, times = _differentiate(values, times)
                if not values.size:
                    return
            magnitudes.extend(float(value) for value in np.linalg.norm(values, axis=1))

        for moment in moments:
            usable = (
                moment.points is not None
                and moment.scores is not None
                and moment.points.shape == (17, 2)
                and moment.scores.shape == (17,)
                and moment.scores[joint] >= min_conf
                and np.isfinite(moment.points[joint]).all()
                and math.isfinite(moment.timestamp)
            )
            if not usable:
                finish_run()
                run_points = []
                run_times = []
                continue
            if run_times and moment.timestamp <= run_times[-1]:
                # A decoder timestamp reset is a discontinuity, not a negative
                # time step that may be discarded and differentiated across.
                finish_run()
                run_points = []
                run_times = []
            run_points.append(moment.points[joint])
            run_times.append(moment.timestamp)
        finish_run()
    return np.asarray(magnitudes, dtype=np.float64)


def _robust_pose_range(
    moments: Sequence[_PoseMoment], joints: Sequence[int], min_conf: float
) -> Optional[float]:
    """Median joint-wise 10th-to-90th percentile coordinate-span norm."""
    ranges: List[float] = []
    for joint in joints:
        points = [
            moment.points[joint]
            for moment in moments
            if moment.points is not None
            and moment.scores is not None
            and moment.scores[joint] >= min_conf
            and np.isfinite(moment.points[joint]).all()
        ]
        if len(points) < 2:
            continue
        array = np.asarray(points, dtype=np.float64)
        low, high = np.percentile(array, (10.0, 90.0), axis=0)
        ranges.append(float(np.linalg.norm(high - low)))
    return float(np.median(ranges)) if ranges else None


def _transition_speeds(
    moments: Sequence[_PoseMoment], joints: Sequence[int], min_conf: float
) -> np.ndarray:
    """Median joint speed per observed adjacent transition."""
    speeds: List[float] = []
    for previous, current in zip(moments, moments[1:]):
        if (
            previous.points is None
            or previous.scores is None
            or current.points is None
            or current.scores is None
        ):
            continue
        dt = current.timestamp - previous.timestamp
        if not math.isfinite(dt) or dt <= 0.0:
            continue
        values = [
            float(np.linalg.norm(current.points[joint] - previous.points[joint]) / dt)
            for joint in joints
            if previous.scores[joint] >= min_conf
            and current.scores[joint] >= min_conf
            and np.isfinite(previous.points[joint]).all()
            and np.isfinite(current.points[joint]).all()
        ]
        if values:
            speeds.append(float(np.median(values)))
    return np.asarray(speeds, dtype=np.float64)


def _symmetry_index(moments: Sequence[_PoseMoment], min_conf: float) -> Optional[float]:
    """Within-clip left/right speed agreement in ``[0, 1]``."""
    left = _joint_derivative_magnitudes(moments, _LEFT_JOINTS, 1, min_conf)
    right = _joint_derivative_magnitudes(moments, _RIGHT_JOINTS, 1, min_conf)
    if not left.size or not right.size:
        return None
    left_median = float(np.median(left))
    right_median = float(np.median(right))
    total = left_median + right_median
    if total <= 1e-12:
        return 1.0
    return float(1.0 - abs(left_median - right_median) / total)


def _safe_ratio(generated: Optional[float], reference: Optional[float]) -> Optional[float]:
    """Directional generated/reference ratio without epsilon distortion."""
    if generated is None or reference is None:
        return None
    if not math.isfinite(generated) or not math.isfinite(reference):
        return None
    if abs(reference) <= 1e-12:
        return 1.0 if abs(generated) <= 1e-12 else None
    return float(max(0.0, generated / reference))


class BodyMotionKinematicsModule(PipelineModule):
    """Compare reference-relative body-motion summary diagnostics."""

    name = "body_motion_kinematics"
    description = "Reference-relative 2D body-motion kinematic diagnostics without frame alignment"
    default_config = {
        "device": "auto",
        "models_dir": "models",
        "moments": 64,
        "min_conf": 0.3,
        "min_derivative_samples": 8,
        "idle_speed_threshold": 0.05,
    }
    models = [
        {
            "id": "yolox_m.onnx",
            "type": "local",
            "task": "single prominent-person detection",
            "auto_download": True,
            "url": (
                "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/"
                "rtmpose_fidelity/yolox_m.onnx"
            ),
            "notes": "Loaded through the shared ayase.pose backend",
        },
        {
            "id": "rtmpose_m.onnx",
            "type": "local",
            "task": "COCO-17 2D pose estimation",
            "auto_download": True,
            "url": (
                "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/"
                "rtmpose_fidelity/rtmpose_m.onnx"
            ),
            "notes": "Loaded through the shared ayase.pose backend",
        },
    ]
    metric_info = {
        "body_motion_speed_ratio": "Median normalized speed, generated/reference (1=equal)",
        "body_motion_acceleration_ratio": (
            "Median normalized acceleration, generated/reference (1=equal)"
        ),
        "body_motion_jerk_ratio": "Median normalized jerk, generated/reference (1=equal)",
        "body_motion_range_ratio": "Robust normalized pose range, generated/reference (1=equal)",
        "body_motion_left_right_symmetry_difference": (
            "Absolute generated-reference difference in left/right speed symmetry (0=equal)"
        ),
        "body_motion_idle_fraction_difference": (
            "Absolute generated-reference difference in idle-transition fraction (0=equal)"
        ),
        "body_motion_pose_coverage": (
            "Minimum generated/reference usable-pose coverage (0-1, higher=more observable)"
        ),
        "body_motion_arm_coverage": (
            "Minimum generated/reference coverage with at least one complete arm chain "
            "(0-1, higher=more observable)"
        ),
    }
    metric_groups = {field: "motion" for field in metric_info}

    def __init__(self, config=None):
        super().__init__(config)
        self.moments = max(2, int(self.config.get("moments", 64)))
        self.min_conf = float(self.config.get("min_conf", 0.3))
        self.min_derivative_samples = max(
            1, int(self.config.get("min_derivative_samples", 8))
        )
        self.idle_speed_threshold = max(
            0.0, float(self.config.get("idle_speed_threshold", 0.05))
        )
        self._backend = None

    def setup(self) -> None:
        """Load the shared detector/pose backend on demand."""
        from ayase.pose import load_pose_backend

        self._backend = load_pose_backend(
            device=self.config.get("device", "auto"),
            models_dir=self.config.get("models_dir", "models"),
        )
        if self._backend is None:
            logger.warning("body_motion_kinematics: pose backend unavailable; metric disabled")

    def process(self, sample: Sample) -> Sample:
        """Attach diagnostics when both videos and the pose backend are available."""
        if self._backend is None or not sample.is_video or sample.reference_path is None:
            return sample
        reference = Path(sample.reference_path)
        if not Path(sample.path).is_file() or not reference.is_file():
            return sample
        try:
            result = self._compare(Path(sample.path), reference)
        except Exception as exc:  # pragma: no cover - decoder/backend specific
            logger.warning("body_motion_kinematics failed for %s: %s", sample.path, exc)
            return sample
        if result is None:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        qm = sample.quality_metrics
        qm.body_motion_speed_ratio = result.get("body_motion_speed_ratio")
        qm.body_motion_acceleration_ratio = result.get("body_motion_acceleration_ratio")
        qm.body_motion_jerk_ratio = result.get("body_motion_jerk_ratio")
        qm.body_motion_range_ratio = result.get("body_motion_range_ratio")
        qm.body_motion_left_right_symmetry_difference = result.get(
            "body_motion_left_right_symmetry_difference"
        )
        qm.body_motion_idle_fraction_difference = result.get(
            "body_motion_idle_fraction_difference"
        )
        qm.body_motion_pose_coverage = result.get("body_motion_pose_coverage")
        qm.body_motion_arm_coverage = result.get("body_motion_arm_coverage")
        return sample

    def _track(self, path: Path) -> _ClipTrack:
        """Extract normalized poses while retaining requested-frame gaps."""
        import cv2

        from ayase.pose import body_origin, body_scale, pose_keypoints

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return _ClipTrack((), 0.0, 0.0, 0)
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if frame_count <= 1 or not math.isfinite(fps) or fps <= 0.0:
                return _ClipTrack((), 0.0, 0.0, 0)
            indices = np.unique(
                np.linspace(0, frame_count - 1, min(self.moments, frame_count), dtype=int)
            )
            moments: List[_PoseMoment] = []
            found = 0
            arms = 0
            previous_timestamp = -math.inf
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                decoder_time = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                fallback_time = float(index) / fps
                timestamp = decoder_time
                if (
                    not math.isfinite(timestamp)
                    or timestamp < 0.0
                    or timestamp <= previous_timestamp
                ):
                    timestamp = fallback_time
                previous_timestamp = timestamp
                if not ok:
                    moments.append(_PoseMoment(timestamp, None, None))
                    continue
                people = pose_keypoints(frame, backend=self._backend)
                if not people:
                    moments.append(_PoseMoment(timestamp, None, None))
                    continue
                person = people[0]
                points = np.asarray(person["keypoints"], dtype=np.float64)
                scores = np.asarray(person["scores"], dtype=np.float64)
                if points.shape != (17, 2) or scores.shape != (17,):
                    moments.append(_PoseMoment(timestamp, None, None))
                    continue
                scale = body_scale(points, scores, self.min_conf)
                origin = body_origin(points, scores, self.min_conf)
                if scale is None or scale <= 1e-12 or origin is None:
                    moments.append(_PoseMoment(timestamp, None, None))
                    continue
                normalized = (points - origin) / scale
                if not np.isfinite(normalized).all():
                    moments.append(_PoseMoment(timestamp, None, None))
                    continue
                found += 1
                if any(all(scores[joint] >= self.min_conf for joint in chain) for chain in _ARM_CHAINS):
                    arms += 1
                moments.append(_PoseMoment(timestamp, normalized, scores))
            attempted = len(indices)
            return _ClipTrack(
                tuple(moments),
                found / attempted if attempted else 0.0,
                arms / attempted if attempted else 0.0,
                attempted,
            )
        finally:
            cap.release()

    def _clip_summary(self, track: _ClipTrack) -> Dict[str, Optional[float]]:
        """Compute independent summary statistics for one clip."""
        result: Dict[str, Optional[float]] = {}
        for name, order in (("speed", 1), ("acceleration", 2), ("jerk", 3)):
            values = _joint_derivative_magnitudes(
                track.moments, _KINEMATIC_JOINTS, order, self.min_conf
            )
            result[name] = (
                float(np.median(values))
                if len(values) >= self.min_derivative_samples
                else None
            )
        result["range"] = _robust_pose_range(track.moments, _KINEMATIC_JOINTS, self.min_conf)
        result["symmetry"] = _symmetry_index(track.moments, self.min_conf)
        transitions = _transition_speeds(track.moments, _KINEMATIC_JOINTS, self.min_conf)
        result["idle_fraction"] = (
            float(np.mean(transitions <= self.idle_speed_threshold))
            if transitions.size
            else None
        )
        return result

    def _compare(self, generated: Path, reference: Path) -> Optional[Dict[str, float]]:
        """Compare independently summarized clips; no moment pairing occurs."""
        generated_track = self._track(generated)
        reference_track = self._track(reference)
        if generated_track.attempted <= 0 or reference_track.attempted <= 0:
            return None
        generated_summary = self._clip_summary(generated_track)
        reference_summary = self._clip_summary(reference_track)
        result: Dict[str, float] = {
            "body_motion_pose_coverage": round(
                min(generated_track.pose_coverage, reference_track.pose_coverage), 6
            ),
            "body_motion_arm_coverage": round(
                min(generated_track.arm_coverage, reference_track.arm_coverage), 6
            ),
        }
        for statistic, field in (
            ("speed", "body_motion_speed_ratio"),
            ("acceleration", "body_motion_acceleration_ratio"),
            ("jerk", "body_motion_jerk_ratio"),
            ("range", "body_motion_range_ratio"),
        ):
            ratio = _safe_ratio(generated_summary[statistic], reference_summary[statistic])
            if ratio is not None:
                result[field] = round(ratio, 6)
        generated_symmetry = generated_summary["symmetry"]
        reference_symmetry = reference_summary["symmetry"]
        if generated_symmetry is not None and reference_symmetry is not None:
            result["body_motion_left_right_symmetry_difference"] = round(
                abs(generated_symmetry - reference_symmetry), 6
            )
        generated_idle = generated_summary["idle_fraction"]
        reference_idle = reference_summary["idle_fraction"]
        if generated_idle is not None and reference_idle is not None:
            result["body_motion_idle_fraction_difference"] = round(
                abs(generated_idle - reference_idle), 6
            )
        return result

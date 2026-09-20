"""Deterministic tests for body-motion kinematic diagnostics."""

from pathlib import Path

import numpy as np
import pytest

from ayase.models import Sample
from ayase.modules.body_motion_kinematics import (
    BodyMotionKinematicsModule,
    _ClipTrack,
    _PoseMoment,
    _joint_derivative_magnitudes,
)


def _track(
    times,
    amplitude=1.0,
    *,
    missing=(),
    left_amplitude=None,
    right_amplitude=None,
    coverage=1.0,
    arm_coverage=1.0,
):
    """Synthetic normalized COCO-17 track with polynomial motion."""
    moments = []
    missing = set(missing)
    for index, timestamp in enumerate(times):
        if index in missing:
            moments.append(_PoseMoment(float(timestamp), None, None))
            continue
        points = np.zeros((17, 2), dtype=np.float64)
        scores = np.ones(17, dtype=np.float64)
        for joint in range(17):
            side_amplitude = amplitude
            if joint in (5, 7, 9, 11, 13, 15) and left_amplitude is not None:
                side_amplitude = left_amplitude
            if joint in (6, 8, 10, 12, 14, 16) and right_amplitude is not None:
                side_amplitude = right_amplitude
            phase = 0.015 * joint
            points[joint, 0] = side_amplitude * (timestamp**3 + phase * timestamp)
            points[joint, 1] = side_amplitude * (0.5 * timestamp**2 - phase * timestamp)
        moments.append(_PoseMoment(float(timestamp), points, scores))
    return _ClipTrack(tuple(moments), coverage, arm_coverage, len(times))


def _compare(module, generated, reference, monkeypatch):
    tracks = {Path("generated.mp4"): generated, Path("reference.mp4"): reference}
    monkeypatch.setattr(module, "_track", lambda path: tracks[path])
    return module._compare(Path("generated.mp4"), Path("reference.mp4"))


def test_identity_has_unit_ratios_and_zero_differences(monkeypatch):
    module = BodyMotionKinematicsModule({"min_derivative_samples": 1})
    track = _track(np.linspace(0.0, 2.0, 21), coverage=0.9, arm_coverage=0.8)
    result = _compare(module, track, track, monkeypatch)

    assert result is not None
    assert result["body_motion_speed_ratio"] == pytest.approx(1.0)
    assert result["body_motion_acceleration_ratio"] == pytest.approx(1.0)
    assert result["body_motion_jerk_ratio"] == pytest.approx(1.0)
    assert result["body_motion_range_ratio"] == pytest.approx(1.0)
    assert result["body_motion_left_right_symmetry_difference"] == pytest.approx(0.0)
    assert result["body_motion_idle_fraction_difference"] == pytest.approx(0.0)
    assert result["body_motion_pose_coverage"] == pytest.approx(0.9)
    assert result["body_motion_arm_coverage"] == pytest.approx(0.8)


def test_derivatives_use_seconds_and_are_sampling_rate_invariant():
    slow_sampling = _track(np.linspace(0.0, 2.0, 21))
    fast_sampling = _track(np.linspace(0.0, 2.0, 41))
    module = BodyMotionKinematicsModule({"min_derivative_samples": 1})
    slow = module._clip_summary(slow_sampling)
    fast = module._clip_summary(fast_sampling)

    assert slow["speed"] == pytest.approx(fast["speed"], rel=0.04)
    assert slow["acceleration"] == pytest.approx(fast["acceleration"], rel=0.06)
    assert slow["jerk"] == pytest.approx(fast["jerk"], rel=1e-10)


def test_amplitude_scaling_preserves_ratio_direction(monkeypatch):
    times = np.linspace(0.0, 2.0, 21)
    generated = _track(times, amplitude=2.0)
    reference = _track(times, amplitude=1.0)
    module = BodyMotionKinematicsModule({"min_derivative_samples": 1})
    result = _compare(module, generated, reference, monkeypatch)

    assert result is not None
    assert result["body_motion_speed_ratio"] == pytest.approx(2.0)
    assert result["body_motion_acceleration_ratio"] == pytest.approx(2.0)
    assert result["body_motion_jerk_ratio"] == pytest.approx(2.0)
    assert result["body_motion_range_ratio"] == pytest.approx(2.0)


def test_missing_detection_splits_runs_instead_of_bridging_gap():
    scores = np.ones(17, dtype=np.float64)
    before = np.zeros((17, 2), dtype=np.float64)
    after = np.full((17, 2), 1000.0, dtype=np.float64)
    moments = (
        _PoseMoment(0.0, before, scores),
        _PoseMoment(1.0, before + 1.0, scores),
        _PoseMoment(2.0, None, None),
        _PoseMoment(3.0, after, scores),
        _PoseMoment(4.0, after + 1.0, scores),
    )

    speeds = _joint_derivative_magnitudes(moments, (0,), 1, 0.3)

    assert speeds.tolist() == pytest.approx([np.sqrt(2.0), np.sqrt(2.0)])


def test_symmetry_difference_detects_one_sided_motion(monkeypatch):
    times = np.linspace(0.0, 2.0, 21)
    reference = _track(times, left_amplitude=1.0, right_amplitude=1.0)
    generated = _track(times, left_amplitude=2.0, right_amplitude=0.0)
    module = BodyMotionKinematicsModule({"min_derivative_samples": 1})
    result = _compare(module, generated, reference, monkeypatch)

    assert result is not None
    assert result["body_motion_left_right_symmetry_difference"] > 0.99


def test_range_ratio_keeps_generated_reference_direction(monkeypatch):
    times = np.linspace(0.0, 2.0, 21)
    module = BodyMotionKinematicsModule({"min_derivative_samples": 1})
    larger = _compare(module, _track(times, 3.0), _track(times, 1.0), monkeypatch)
    smaller = _compare(module, _track(times, 1.0), _track(times, 2.0), monkeypatch)

    assert larger is not None and smaller is not None
    assert larger["body_motion_range_ratio"] == pytest.approx(3.0)
    assert smaller["body_motion_range_ratio"] == pytest.approx(0.5)


def test_pair_coverage_uses_lower_clip_coverage(monkeypatch):
    times = np.linspace(0.0, 1.0, 8)
    generated = _track(times, coverage=0.75, arm_coverage=0.5)
    reference = _track(times, coverage=0.9, arm_coverage=0.25)
    module = BodyMotionKinematicsModule({"min_derivative_samples": 1})
    result = _compare(module, generated, reference, monkeypatch)

    assert result is not None
    assert result["body_motion_pose_coverage"] == pytest.approx(0.75)
    assert result["body_motion_arm_coverage"] == pytest.approx(0.25)


def test_idle_fraction_difference_reports_temporal_stillness(monkeypatch):
    times = np.linspace(0.0, 2.0, 21)
    moving = _track(times)
    partly_idle_moments = []
    for moment in moving.moments:
        assert moment.points is not None
        points = moment.points.copy()
        if moment.timestamp <= 1.0:
            points[:] = 0.0
        partly_idle_moments.append(_PoseMoment(moment.timestamp, points, moment.scores))
    partly_idle = _ClipTrack(tuple(partly_idle_moments), 1.0, 1.0, len(times))
    module = BodyMotionKinematicsModule(
        {"min_derivative_samples": 1, "idle_speed_threshold": 0.01}
    )

    result = _compare(module, partly_idle, moving, monkeypatch)

    assert result is not None
    assert result["body_motion_idle_fraction_difference"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "sample",
    [
        Sample(path=Path("image.png"), is_video=False, reference_path=Path("reference.png")),
        Sample(path=Path("video.mp4"), is_video=True),
    ],
)
def test_process_is_noop_without_applicable_inputs(sample):
    module = BodyMotionKinematicsModule()
    module._backend = object()

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics is None


def test_process_is_noop_without_backend():
    sample = Sample(
        path=Path("video.mp4"), is_video=True, reference_path=Path("reference.mp4")
    )

    result = BodyMotionKinematicsModule().process(sample)

    assert result is sample
    assert result.quality_metrics is None


def test_process_catches_backend_or_decoder_failure(tmp_path, monkeypatch):
    generated = tmp_path / "generated.mp4"
    reference = tmp_path / "reference.mp4"
    generated.touch()
    reference.touch()
    sample = Sample(path=generated, is_video=True, reference_path=reference)
    module = BodyMotionKinematicsModule()
    module._backend = object()

    def fail(*_args):
        raise RuntimeError("synthetic decoder failure")

    monkeypatch.setattr(module, "_compare", fail)

    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_process_attaches_each_diagnostic_and_returns_same_sample(tmp_path, monkeypatch):
    generated = tmp_path / "generated.mp4"
    reference = tmp_path / "reference.mp4"
    generated.touch()
    reference.touch()
    sample = Sample(path=generated, is_video=True, reference_path=reference)
    module = BodyMotionKinematicsModule()
    module._backend = object()
    expected = {
        "body_motion_speed_ratio": 1.1,
        "body_motion_acceleration_ratio": 1.2,
        "body_motion_jerk_ratio": 1.3,
        "body_motion_range_ratio": 0.9,
        "body_motion_left_right_symmetry_difference": 0.1,
        "body_motion_idle_fraction_difference": 0.2,
        "body_motion_pose_coverage": 0.8,
        "body_motion_arm_coverage": 0.7,
    }
    monkeypatch.setattr(module, "_compare", lambda *_args: expected)

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics is not None
    for field, value in expected.items():
        assert getattr(result.quality_metrics, field) == pytest.approx(value)


def test_metadata_documents_all_transparent_outputs():
    metadata = BodyMotionKinematicsModule.get_metadata()

    assert metadata["name"] == "body_motion_kinematics"
    assert metadata["input_type"] == "vid +ref"
    assert set(metadata["output_fields"]) == set(BodyMotionKinematicsModule.metric_info)
    assert "similarity" not in metadata["output_fields"]
    assert set(BodyMotionKinematicsModule.metric_groups.values()) == {"motion"}

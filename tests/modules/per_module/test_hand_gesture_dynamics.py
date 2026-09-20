"""Deterministic tests for 2D hand/finger dynamics diagnostics."""

from pathlib import Path

import numpy as np
import pytest

from ayase.models import Sample
from ayase.modules.hand_gesture_dynamics import (
    BODY_SLICE,
    HAND_JOINTS,
    LEFT_HAND_SLICE,
    RIGHT_HAND_SLICE,
    HandGestureDynamicsModule,
    HandMoment,
    HandTrack,
    _shape_speeds,
    compare_hand_tracks,
    hand_descriptors,
    normalize_hand,
    summarize_hand_track,
)


def _base_hand():
    points = np.asarray(
        [
            [0.0, 0.0],
            [-0.35, 0.30],
            [-0.60, 0.60],
            [-0.85, 0.90],
            [-1.05, 1.20],
            [-0.45, 0.90],
            [-0.45, 1.40],
            [-0.45, 1.90],
            [-0.45, 2.40],
            [0.00, 1.00],
            [0.00, 1.60],
            [0.00, 2.20],
            [0.00, 2.80],
            [0.40, 0.90],
            [0.40, 1.40],
            [0.40, 1.90],
            [0.40, 2.40],
            [0.78, 0.72],
            [0.82, 1.12],
            [0.86, 1.52],
            [0.90, 1.92],
        ],
        dtype=np.float64,
    )
    return points, np.ones(HAND_JOINTS, dtype=bool)


def _moving_hand(timestamp, motion):
    points, valid = _base_hand()
    # Distal-only motion changes hand shape, not wrist workspace.
    points[[3, 4], 0] -= motion * timestamp
    points[[7, 8, 11, 12, 15, 16, 19, 20], 0] += motion * timestamp
    return points, valid


def _track(
    times,
    *,
    left_motion=0.2,
    right_motion=0.2,
    missing=(),
    coverage=1.0,
    observability=1.0,
):
    moments = []
    missing = set(missing)
    for index, timestamp in enumerate(times):
        if index in missing:
            moments.append(HandMoment(float(timestamp), None, None, None, None))
            continue
        left, left_valid = _moving_hand(timestamp, left_motion)
        right, right_valid = _moving_hand(timestamp, right_motion)
        moments.append(
            HandMoment(
                float(timestamp), left, left_valid, right, right_valid
            )
        )
    return HandTrack(tuple(moments), coverage, observability, len(times))


def test_module_basics():
    from tests.modules.conftest import _test_module_basics

    _test_module_basics(HandGestureDynamicsModule, "hand_gesture_dynamics")


def test_native_wholebody_hand_indices_are_explicit():
    assert BODY_SLICE == slice(0, 17)
    assert LEFT_HAND_SLICE == slice(91, 112)
    assert RIGHT_HAND_SLICE == slice(112, 133)


def test_palm_normalization_is_translation_and_scale_invariant():
    canonical, _ = _base_hand()
    scores = np.ones(HAND_JOINTS)
    first = normalize_hand(canonical * 30.0 + [200.0, 100.0], scores, 0.3)
    second = normalize_hand(canonical * 75.0 + [-20.0, 350.0], scores, 0.3)

    assert first is not None and second is not None
    assert first[0] == pytest.approx(second[0])
    assert first[1].tolist() == second[1].tolist()
    assert first[0][0].tolist() == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize("low_indices", [(0,), (5, 9), (5, 9, 13)])
def test_palm_normalization_rejects_unusable_scale(low_indices):
    points, _ = _base_hand()
    scores = np.ones(HAND_JOINTS)
    scores[list(low_indices)] = 0.0

    assert normalize_hand(points, scores, 0.3) is None


def test_descriptors_have_documented_geometry():
    points, valid = _base_hand()
    descriptors = hand_descriptors(points, valid)

    assert descriptors.articulation == pytest.approx(1.0)
    assert descriptors.openness is not None and descriptors.openness > 2.0
    assert descriptors.pinch == pytest.approx(np.linalg.norm(points[4] - points[8]))


def test_descriptor_is_unset_when_required_joints_are_not_observable():
    points, valid = _base_hand()
    valid[[1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20]] = False

    descriptors = hand_descriptors(points, valid)

    assert descriptors.articulation is None
    assert descriptors.openness is None
    assert descriptors.pinch is None


def test_identical_unaligned_tracks_have_zero_differences():
    track = _track(np.linspace(0.0, 2.0, 12), coverage=0.75, observability=0.6)

    result = compare_hand_tracks(track, track, min_samples=2, min_speed_samples=2)

    assert result["hand_gesture_sample_coverage"] == pytest.approx(0.75)
    assert result["hand_gesture_reference_coverage"] == pytest.approx(0.75)
    assert result["hand_gesture_sample_joint_observability"] == pytest.approx(0.6)
    assert result["hand_gesture_reference_joint_observability"] == pytest.approx(0.6)
    for field, value in result.items():
        if field.endswith("difference"):
            assert value == pytest.approx(0.0)


def test_location_and_amplitude_are_independent_diagnostics():
    times = np.linspace(0.0, 2.0, 15)
    reference = _track(times, left_motion=0.05, right_motion=0.05)
    sample = _track(times, left_motion=0.8, right_motion=0.8)

    result = compare_hand_tracks(sample, reference, min_samples=2, min_speed_samples=2)

    assert result["hand_gesture_openness_location_difference"] > 0.0
    assert result["hand_gesture_openness_amplitude_difference"] > 0.0
    assert result["hand_gesture_pinch_location_difference"] > 0.0
    assert result["hand_gesture_pinch_amplitude_difference"] > 0.0


def test_speed_uses_seconds_and_is_sampling_rate_invariant():
    slow = _track(np.linspace(0.0, 2.0, 21))
    fast = _track(np.linspace(0.0, 2.0, 41))

    slow_summary = summarize_hand_track(
        slow, min_samples=2, min_speed_samples=2, min_velocity_joints=8
    )
    fast_summary = summarize_hand_track(
        fast, min_samples=2, min_speed_samples=2, min_velocity_joints=8
    )

    assert slow_summary["speed"] == pytest.approx(fast_summary["speed"], rel=1e-10)


def test_missing_frame_breaks_speed_run_instead_of_bridging_jump():
    base, valid = _base_hand()
    before = base.copy()
    before_next = base.copy()
    before_next[1:, 0] += 1.0
    after = base.copy()
    after[1:, 0] += 1000.0
    after_next = after.copy()
    after_next[1:, 0] += 1.0
    moments = (
        HandMoment(0.0, before, valid, None, None),
        HandMoment(1.0, before_next, valid, None, None),
        HandMoment(2.0, None, None, None, None),
        HandMoment(3.0, after, valid, None, None),
        HandMoment(4.0, after_next, valid, None, None),
    )

    speeds = _shape_speeds(moments, "left", min_velocity_joints=8)

    assert speeds.tolist() == pytest.approx([1.0, 1.0])


def test_left_right_asymmetry_difference_detects_one_sided_dynamics():
    times = np.linspace(0.0, 2.0, 21)
    reference = _track(times, left_motion=0.4, right_motion=0.4)
    sample = _track(times, left_motion=0.8, right_motion=0.0)

    result = compare_hand_tracks(sample, reference, min_samples=2, min_speed_samples=2)

    assert result["hand_gesture_left_right_asymmetry_difference"] > 0.99


def test_coverage_is_retained_when_descriptors_lack_samples():
    short = _track([0.0], coverage=0.5, observability=0.25)
    full = _track(np.linspace(0.0, 1.0, 8))

    result = compare_hand_tracks(short, full, min_samples=4, min_speed_samples=3)

    assert result["hand_gesture_sample_coverage"] == pytest.approx(0.5)
    assert result["hand_gesture_sample_joint_observability"] == pytest.approx(0.25)
    assert result["hand_gesture_articulation_location_difference"] is None
    assert result["hand_gesture_articulation_speed_difference"] is None


@pytest.mark.parametrize(
    "sample",
    [
        Sample(path=Path("image.png"), is_video=False, reference_path=Path("reference.png")),
        Sample(path=Path("video.mp4"), is_video=True),
    ],
)
def test_process_is_noop_without_applicable_inputs(sample):
    module = HandGestureDynamicsModule()
    module._wholebody = object()

    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_process_is_noop_without_backend(tmp_path):
    generated = tmp_path / "generated.mp4"
    reference = tmp_path / "reference.mp4"
    generated.touch()
    reference.touch()
    sample = Sample(path=generated, is_video=True, reference_path=reference)

    assert HandGestureDynamicsModule().process(sample) is sample
    assert sample.quality_metrics is None


def test_process_catches_backend_or_decoder_failure(tmp_path, monkeypatch):
    generated = tmp_path / "generated.mp4"
    reference = tmp_path / "reference.mp4"
    generated.touch()
    reference.touch()
    sample = Sample(path=generated, is_video=True, reference_path=reference)
    module = HandGestureDynamicsModule()
    module._wholebody = object()

    def fail(*_args):
        raise RuntimeError("synthetic decoder failure")

    monkeypatch.setattr(module, "_compare", fail)

    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_process_catches_backend_construction_failure(tmp_path, monkeypatch):
    generated = tmp_path / "generated.mp4"
    reference = tmp_path / "reference.mp4"
    generated.touch()
    reference.touch()
    sample = Sample(path=generated, is_video=True, reference_path=reference)
    module = HandGestureDynamicsModule()

    def fail():
        raise RuntimeError("synthetic model failure")

    monkeypatch.setattr(module, "_get_backend", fail)

    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_process_attaches_only_available_diagnostics(tmp_path, monkeypatch):
    generated = tmp_path / "generated.mp4"
    reference = tmp_path / "reference.mp4"
    generated.touch()
    reference.touch()
    sample = Sample(path=generated, is_video=True, reference_path=reference)
    module = HandGestureDynamicsModule()
    module._wholebody = object()
    result = {field: index / 100.0 for index, field in enumerate(module.metric_info, 1)}
    result["hand_gesture_pinch_amplitude_difference"] = None
    monkeypatch.setattr(module, "_compare", lambda *_args: result)

    assert module.process(sample) is sample
    assert sample.quality_metrics is not None
    for field, value in result.items():
        if value is None:
            assert getattr(sample.quality_metrics, field) is None
            assert field not in sample.quality_metrics.metric_backends
        else:
            assert getattr(sample.quality_metrics, field) == pytest.approx(value)
            assert sample.quality_metrics.metric_backends[field] == module._backend_label


def test_metadata_documents_only_transparent_diagnostics():
    metadata = HandGestureDynamicsModule.get_metadata()

    assert metadata["name"] == "hand_gesture_dynamics"
    assert metadata["input_type"] == "vid +ref"
    assert set(metadata["output_fields"]) == set(HandGestureDynamicsModule.metric_info)
    assert not any("similarity" in field for field in metadata["output_fields"])
    assert set(HandGestureDynamicsModule.metric_groups.values()) == {"motion"}

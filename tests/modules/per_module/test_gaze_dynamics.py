"""Deterministic tests for MediaPipe eye-look activation diagnostics."""

from pathlib import Path

import numpy as np
import pytest

from ayase.models import Sample
from ayase.modules._blendshape_utils import (
    BLENDSHAPE_DIM,
    CANONICAL_BLENDSHAPES,
    BlendshapeTrajectory,
)
from ayase.modules.gaze_dynamics import (
    GazeDynamicsModule,
    compare_ocular_trajectories,
    summarize_ocular_trajectory,
)

_INDEX = {name: index for index, name in enumerate(CANONICAL_BLENDSHAPES)}


def _set_signed(row, positive_name, negative_name, value):
    row[_INDEX[positive_name]] = max(0.0, value)
    row[_INDEX[negative_name]] = max(0.0, -value)


def _trajectory(
    horizontal,
    vertical=None,
    *,
    right_horizontal=None,
    right_vertical=None,
    valid=None,
    fps=10.0,
    frame_indices=None,
):
    horizontal = np.asarray(horizontal, dtype=np.float64)
    count = len(horizontal)
    vertical = np.zeros(count) if vertical is None else np.asarray(vertical, dtype=np.float64)
    right_horizontal = (
        horizontal if right_horizontal is None else np.asarray(right_horizontal, dtype=np.float64)
    )
    right_vertical = (
        vertical if right_vertical is None else np.asarray(right_vertical, dtype=np.float64)
    )
    coefficients = np.zeros((count, BLENDSHAPE_DIM), dtype=np.float32)
    for index in range(count):
        _set_signed(
            coefficients[index], "eyeLookOutLeft", "eyeLookInLeft", horizontal[index]
        )
        _set_signed(
            coefficients[index],
            "eyeLookInRight",
            "eyeLookOutRight",
            right_horizontal[index],
        )
        _set_signed(
            coefficients[index], "eyeLookUpLeft", "eyeLookDownLeft", vertical[index]
        )
        _set_signed(
            coefficients[index],
            "eyeLookUpRight",
            "eyeLookDownRight",
            right_vertical[index],
        )
    validity = np.ones(count, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    indices = (
        np.arange(count, dtype=np.int64)
        if frame_indices is None
        else np.asarray(frame_indices, dtype=np.int64)
    )
    timestamps = indices.astype(np.float64) / fps
    return BlendshapeTrajectory(
        timestamps_sec=timestamps,
        coefficients=coefficients,
        valid=validity,
        frame_indices=indices,
        fps=fps,
        decoded_frames=count,
        face_frames=int(validity.sum()),
    )


def test_module_basics():
    from tests.modules.conftest import _test_module_basics

    _test_module_basics(GazeDynamicsModule, "gaze_dynamics")


def test_identical_unaligned_trajectories_have_zero_differences():
    values = np.asarray([-0.4, -0.2, 0.0, 0.2, 0.4, 0.2, 0.0, -0.2])
    track = _trajectory(values, values * 0.5)

    result = compare_ocular_trajectories(track, track, min_samples=2)

    for field, value in result.items():
        if field.endswith("coverage"):
            assert value == pytest.approx(1.0)
        else:
            assert value == pytest.approx(0.0)


def test_location_and_amplitude_are_reported_separately():
    reference = _trajectory(np.linspace(-0.2, 0.2, 11), np.linspace(-0.1, 0.1, 11))
    sample = _trajectory(np.linspace(0.1, 0.9, 11), np.linspace(-0.3, 0.1, 11))

    result = compare_ocular_trajectories(sample, reference, min_samples=2)

    assert result["gaze_blendshape_horizontal_location_difference"] == pytest.approx(0.5)
    assert result["gaze_blendshape_vertical_location_difference"] == pytest.approx(0.1)
    assert result["gaze_blendshape_horizontal_amplitude_difference"] == pytest.approx(0.32)
    assert result["gaze_blendshape_vertical_amplitude_difference"] == pytest.approx(0.16)


def test_speed_is_per_second_and_sampling_rate_invariant():
    slow_times = np.arange(11) / 10.0
    fast_times = np.arange(21) / 20.0
    slow = _trajectory(0.4 * slow_times, -0.3 * slow_times, fps=10.0)
    fast = _trajectory(0.4 * fast_times, -0.3 * fast_times, fps=20.0)

    slow_summary, _ = summarize_ocular_trajectory(slow, min_samples=2)
    fast_summary, _ = summarize_ocular_trajectory(fast, min_samples=2)

    assert slow_summary is not None and fast_summary is not None
    assert slow_summary.speed == pytest.approx(0.5, rel=1e-6)
    assert fast_summary.speed == pytest.approx(0.5, rel=1e-6)


def test_invalid_frame_breaks_speed_run_instead_of_bridging_gap():
    track = _trajectory(
        [0.0, 0.1, 0.0, 0.9, 1.0],
        valid=[True, True, False, True, True],
        fps=10.0,
    )

    summary, coverage = summarize_ocular_trajectory(track, min_samples=2)

    assert summary is not None
    assert summary.speed == pytest.approx(1.0)
    assert coverage == pytest.approx(0.8)


def test_binocular_disagreement_is_not_conflated_with_mean_direction():
    reference = _trajectory(np.zeros(8))
    sample = _trajectory(np.full(8, 0.3), right_horizontal=np.full(8, -0.3))

    result = compare_ocular_trajectories(sample, reference, min_samples=2)

    assert result["gaze_blendshape_horizontal_location_difference"] == pytest.approx(0.0)
    assert result["gaze_blendshape_binocular_disagreement_difference"] == pytest.approx(0.6)


def test_low_coverage_is_retained_when_summaries_are_unavailable():
    sample = _trajectory(np.zeros(5), valid=[True, False, False, False, False])
    reference = _trajectory(np.zeros(5))

    result = compare_ocular_trajectories(sample, reference, min_samples=3)

    assert result["gaze_blendshape_sample_coverage"] == pytest.approx(0.2)
    assert result["gaze_blendshape_reference_coverage"] == pytest.approx(1.0)
    assert result["gaze_blendshape_speed_difference"] is None
    assert result["gaze_blendshape_horizontal_location_difference"] is None


def test_malformed_or_out_of_range_required_coefficients_are_not_used():
    malformed = _trajectory(np.zeros(4))
    malformed.coefficients[0, _INDEX["eyeLookUpLeft"]] = np.nan
    malformed.coefficients[1, _INDEX["eyeLookUpLeft"]] = 1.2

    summary, coverage = summarize_ocular_trajectory(malformed, min_samples=3)

    assert summary is None
    assert coverage == pytest.approx(0.5)


@pytest.mark.parametrize(
    "sample",
    [
        Sample(path=Path("image.png"), is_video=False, reference_path=Path("reference.png")),
        Sample(path=Path("video.mp4"), is_video=True),
    ],
)
def test_process_is_noop_without_applicable_inputs(sample):
    module = GazeDynamicsModule()
    module._available = True

    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_process_is_noop_without_backend():
    sample = Sample(
        path=Path("video.mp4"), is_video=True, reference_path=Path("reference.mp4")
    )

    assert GazeDynamicsModule().process(sample) is sample
    assert sample.quality_metrics is None


def test_process_catches_extraction_failure(monkeypatch):
    sample = Sample(
        path=Path("video.mp4"), is_video=True, reference_path=Path("reference.mp4")
    )
    module = GazeDynamicsModule()
    module._available = True

    def fail(*_args):
        raise RuntimeError("synthetic backend failure")

    monkeypatch.setattr(module, "_compare", fail)

    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_metadata_has_only_transparent_diagnostics():
    metadata = GazeDynamicsModule.get_metadata()

    assert metadata["name"] == "gaze_dynamics"
    assert metadata["input_type"] == "vid +ref"
    assert set(metadata["metric_info"]) == set(GazeDynamicsModule.metric_info)
    assert set(metadata["output_fields"]) == set(GazeDynamicsModule.metric_info)
    assert "gaze_similarity" not in metadata["metric_info"]
    assert set(GazeDynamicsModule.metric_groups.values()) == {"face"}

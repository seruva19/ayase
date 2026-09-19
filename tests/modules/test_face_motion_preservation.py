"""Focused offline tests for frame-aligned facial-motion preservation metrics."""

from pathlib import Path

import numpy as np
import pytest

from tests.modules.conftest import _test_module_basics


def _landmarks(frames: int = 80) -> np.ndarray:
    """Return 51 non-rigid trajectories with defined x/y correlations."""
    time = np.linspace(-1.0, 1.0, frames, dtype=np.float64)
    points = np.empty((frames, 51, 2), dtype=np.float64)
    for landmark in range(51):
        points[:, landmark, 0] = 0.02 * landmark + (0.003 + 0.0002 * landmark) * time
        points[:, landmark, 1] = 0.01 * landmark + (0.002 + 0.0003 * landmark) * time**3
    return points


def _trajectory(frames: int = 80, *, fps: float = 25.0):
    from ayase.modules.face_motion_preservation import FaceLandmarkTrajectory

    return FaceLandmarkTrajectory(
        positions=_landmarks(frames),
        ear=0.28 + 0.03 * np.sin(np.linspace(0.0, 4.0 * np.pi, frames)),
        valid=np.ones(frames, dtype=bool),
        fps=fps,
        decoded_frames=frames,
    )


def test_face_motion_preservation_basics():
    from ayase.modules.face_motion_preservation import FaceMotionPreservationModule

    _test_module_basics(FaceMotionPreservationModule, "face_motion_preservation")


def test_header_states_frame_alignment_and_non_identity_boundary():
    import ayase.modules.face_motion_preservation as module

    doc = module.__doc__.lower()
    assert "frame-corresponding" in doc
    assert "not an identity metric" in doc
    assert "not valid for unrelated" in doc


def test_identical_landmark_trajectories_score_one():
    from ayase.modules.face_motion_preservation import landmark_pair_correlations

    landmarks = _landmarks()
    result = landmark_pair_correlations(landmarks, landmarks.copy())

    assert result["face_motion_x_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert result["face_motion_y_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert result["face_motion_cca_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert result["face_motion_landmark_pair_coverage"] == pytest.approx(1.0)


def test_temporal_reversal_lowers_frame_aligned_pearson_scores():
    from ayase.modules.face_motion_preservation import landmark_pair_correlations

    landmarks = _landmarks()
    reversed_result = landmark_pair_correlations(landmarks[::-1], landmarks)

    assert reversed_result["face_motion_x_correlation"] < -0.99
    assert reversed_result["face_motion_y_correlation"] < -0.99


def test_first_cca_is_invariant_to_invertible_coordinate_rotation():
    from ayase.modules.face_motion_preservation import _first_canonical_correlation

    time = np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)
    left = np.column_stack((np.sin(time), np.cos(2.0 * time)))
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    right = left @ rotation

    assert _first_canonical_correlation(left, right) == pytest.approx(1.0, abs=1e-12)


def test_constant_landmark_pair_is_excluded_from_coverage():
    from ayase.modules.face_motion_preservation import landmark_pair_correlations

    landmarks = _landmarks()
    landmarks[:, 1, :] = landmarks[:, 0, :] + np.array([0.1, -0.2])
    result = landmark_pair_correlations(landmarks, landmarks.copy())

    assert result["face_motion_x_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert result["face_motion_y_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert result["face_motion_cca_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert np.isfinite(result["face_motion_landmark_pair_coverage"])
    assert 0.0 < result["face_motion_landmark_pair_coverage"] < 1.0


def test_blink_overlap_precision_recall_and_f1_are_event_based():
    from ayase.modules.face_motion_preservation import blink_overlap_scores

    reference = np.full(100, 0.30)
    generated = np.full(100, 0.30)
    # At 100 fps, six frames are the inclusive 60 ms lower duration bound.
    reference[10:16] = 0.10
    reference[40:48] = 0.10
    generated[12:18] = 0.10  # overlaps only the first reference blink
    generated[70:76] = 0.10  # unmatched false positive

    result = blink_overlap_scores(generated, reference, fps=100.0, threshold=0.243)

    assert result["face_motion_blink_precision"] == pytest.approx(0.5)
    assert result["face_motion_blink_recall"] == pytest.approx(0.5)
    assert result["face_motion_blink_f1"] == pytest.approx(0.5)


def test_missing_all_reference_blinks_is_a_zero_not_an_omitted_failure():
    from ayase.modules.face_motion_preservation import blink_overlap_scores

    reference = np.full(100, 0.30)
    reference[10:16] = 0.10
    generated = np.full(100, 0.30)

    result = blink_overlap_scores(generated, reference, fps=100.0, threshold=0.243)

    assert result["face_motion_blink_precision"] == 0.0
    assert result["face_motion_blink_recall"] == 0.0
    assert result["face_motion_blink_f1"] == 0.0
    assert result["face_motion_blink_reference_events"] == 1
    assert result["face_motion_blink_predicted_events"] == 0


def test_nan_gap_breaks_a_closed_eye_run_into_separate_intervals():
    from ayase.modules.face_motion_preservation import _blink_events

    ear = np.full(40, 0.30)
    ear[10:20] = 0.10
    ear[15] = np.nan

    # The NaN splits a nominal ten-frame closure into runs of five and four;
    # both are shorter than the six-frame minimum at 100 fps.
    assert _blink_events(ear, fps=100.0, threshold=0.243) == []


def test_compare_preserves_detection_gaps_as_blink_event_boundaries():
    from ayase.modules.face_motion_preservation import (
        FaceLandmarkTrajectory,
        FaceMotionPreservationModule,
    )

    frames = 40
    ear = np.full(frames, 0.30)
    ear[10:20] = 0.10
    valid = np.ones(frames, dtype=bool)
    valid[15] = False
    positions = _landmarks(frames)
    positions[~valid] = np.nan
    ear[~valid] = np.nan
    trajectory = FaceLandmarkTrajectory(
        positions=positions,
        ear=ear,
        valid=valid,
        fps=100.0,
        decoded_frames=frames,
    )
    module = FaceMotionPreservationModule(
        {"min_paired_frames": 3, "blink_ear_threshold": 0.243}
    )

    result = module.compare_trajectories(trajectory, trajectory)

    # Removing the missing frame would collapse the two sub-threshold runs into
    # a false nine-frame blink. It must remain a boundary on the source timeline.
    assert result["face_motion_blink_precision"] is None
    assert result["face_motion_blink_recall"] is None
    assert result["face_motion_blink_f1"] is None


def test_blink_scoring_requires_an_explicit_calibrated_threshold():
    module = __import__(
        "ayase.modules.face_motion_preservation", fromlist=["FaceMotionPreservationModule"]
    ).FaceMotionPreservationModule({"min_paired_frames": 3})

    result = module.compare_trajectories(_trajectory(), _trajectory())

    assert result["face_motion_blink_precision"] is None
    assert result["face_motion_blink_recall"] is None
    assert result["face_motion_blink_f1"] is None


@pytest.mark.parametrize(
    ("generated", "reference", "message"),
    [
        (_trajectory(fps=25.06), _trajectory(fps=25.0), "fps differs"),
        (_trajectory(frames=81), _trajectory(frames=80), "frame count differs"),
        (_trajectory(frames=83), _trajectory(frames=80), "frame count differs"),
    ],
)
def test_compare_rejects_non_aligned_videos(generated, reference, message):
    from ayase.modules.face_motion_preservation import FaceMotionPreservationModule

    module = FaceMotionPreservationModule({"min_paired_frames": 3})

    with pytest.raises(ValueError, match=message):
        module.compare_trajectories(generated, reference)


def test_process_without_setup_is_a_graceful_noop():
    from ayase.models import Sample
    from ayase.modules.face_motion_preservation import FaceMotionPreservationModule

    sample = Sample(path=Path("clip.mp4"), is_video=True)
    module = FaceMotionPreservationModule()

    assert module.process(sample) is sample
    assert sample.quality_metrics is None
    assert sample.validation_issues == []


def test_store_writes_all_public_fields_metadata_and_backend():
    from ayase.models import Sample
    from ayase.modules.face_motion_preservation import FaceMotionPreservationModule

    module = FaceMotionPreservationModule()
    module._backend = "test_backend"
    result = {
        "face_motion_x_correlation": 0.91,
        "face_motion_y_correlation": 0.82,
        "face_motion_cca_correlation": 0.93,
        "face_motion_ear_correlation": 0.74,
        "face_motion_blink_precision": 0.80,
        "face_motion_blink_recall": 0.67,
        "face_motion_blink_f1": 0.73,
        "face_motion_landmark_pair_coverage": 0.98,
        "face_motion_frame_coverage": 0.95,
        "face_motion_paired_frames": 76,
        "face_motion_backend": "test_backend",
        "face_motion_multiple_faces": False,
    }
    sample = Sample(path=Path("clip.mp4"), is_video=True)

    module._store(sample, result)

    assert sample.quality_metrics is not None
    for field in module.metric_info:
        assert getattr(sample.quality_metrics, field) == result[field]
        assert sample.quality_metrics.metric_backends[field] == "test_backend"
    assert sample.metadata["face_motion_paired_frames"] == 76
    assert sample.metadata["face_motion_multiple_faces"] is False
    assert sample.validation_issues == []

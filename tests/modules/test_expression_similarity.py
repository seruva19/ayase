"""Tests for the time-free expression-manner similarity metric.

The property that separates this metric from ``expression_following`` is that no
score may depend on when something happened -- only on what happened and how
often. Several tests below exist purely to hold that line.
"""

import json

import numpy as np

from tests.modules.conftest import _test_module_basics


def _trajectory(values, valid=None, fps=10.0):
    from ayase.modules._blendshape_utils import BlendshapeTrajectory

    coefficients = np.asarray(values, dtype=np.float32)
    count = len(coefficients)
    valid_array = np.ones(count, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    coefficients = coefficients.copy()
    coefficients[~valid_array] = np.nan
    return BlendshapeTrajectory(
        timestamps_sec=np.arange(count, dtype=np.float64) / fps,
        coefficients=coefficients,
        valid=valid_array,
        frame_indices=np.arange(count, dtype=np.int64),
        fps=fps,
        decoded_frames=count,
        face_frames=int(valid_array.sum()),
    )


def _wave(frames=60, dim=52, seed=0, amplitude=0.4, phase=0.0):
    """A repeatable expressive trajectory: every coefficient oscillates a little."""
    from ayase.modules._blendshape_utils import BLENDSHAPE_DIM

    rng = np.random.default_rng(seed)
    time = np.arange(frames, dtype=np.float64)
    base = np.zeros((frames, BLENDSHAPE_DIM), dtype=np.float64)
    for channel in range(dim):
        frequency = 0.15 + 0.01 * channel
        offset = 0.5 + 0.1 * rng.standard_normal()
        base[:, channel] = offset + amplitude * np.sin(frequency * time + phase + channel)
    return np.clip(base, 0.0, 1.0)


def _module(**config):
    from ayase.modules.expression_similarity import ExpressionSimilarityModule

    return ExpressionSimilarityModule(config or None)


def test_expression_similarity_basics():
    from ayase.modules.expression_similarity import ExpressionSimilarityModule

    _test_module_basics(ExpressionSimilarityModule, "expression_similarity")


def test_identical_clips_score_one():
    values = _wave()
    result = _module().compare_trajectories(_trajectory(values), _trajectory(values))

    assert result["expression_similarity"] == 1.0
    assert result["expression_similarity_distribution"] == 1.0
    assert result["expression_similarity_coactivation"] == 1.0
    assert result["expression_similarity_dynamics"] == 1.0
    assert result["expression_similarity_range_ratio"] == 1.0
    assert result["expression_similarity_flags"] == []


def test_time_shift_does_not_change_the_score():
    """The defining property: rolling a clip in time is invisible to this metric.

    ``expression_following`` would read the same pair as badly desynchronised.
    Here the two clips contain the same expressions at the same rate, which is
    all the metric is allowed to see.
    """
    values = _wave(frames=80)
    shifted = np.roll(values, shift=23, axis=0)
    result = _module().compare_trajectories(_trajectory(values), _trajectory(shifted))

    assert result["expression_similarity"] > 0.97
    assert result["expression_similarity_dynamics"] > 0.95


def test_clip_length_does_not_change_the_score():
    """Halving one clip leaves the manner unchanged, so the score must survive it."""
    values = _wave(frames=120)
    result = _module().compare_trajectories(_trajectory(values), _trajectory(values[:60]))

    assert result["expression_similarity"] > 0.9


def test_frozen_face_is_caught_by_dynamics_and_range():
    """An adapter that stopped the face is the failure identity metrics cannot see."""
    lively = _wave(frames=60)
    frozen = np.repeat(lively[:1], 60, axis=0)
    result = _module().compare_trajectories(_trajectory(frozen), _trajectory(lively))

    assert result["expression_similarity_dynamics"] < 0.2
    assert result["expression_similarity_range_ratio"] < 0.1
    assert result["expression_similarity"] < 0.75
    # Nothing moved, so no coefficient qualifies as active in both clips.
    assert "too_few_active_coefficients" in result["expression_similarity_flags"]
    assert result["expression_similarity_coactivation"] is None


def test_reversed_co_activation_lowers_that_component_only():
    """Same repertoire and same tempo, opposite pairings: co-activation must drop."""
    from ayase.modules._blendshape_utils import BLENDSHAPE_DIM

    frames = 80
    time = np.arange(frames, dtype=np.float64)
    together = np.full((frames, BLENDSHAPE_DIM), 0.5)
    opposed = np.full((frames, BLENDSHAPE_DIM), 0.5)
    wave = 0.3 * np.sin(0.2 * time)
    for channel in range(BLENDSHAPE_DIM):
        together[:, channel] = 0.5 + wave
        opposed[:, channel] = 0.5 + (wave if channel % 2 == 0 else -wave)

    result = _module().compare_trajectories(_trajectory(opposed), _trajectory(together))

    assert result["expression_similarity_coactivation"] < 0.6
    assert result["expression_similarity_distribution"] > 0.95
    assert result["expression_similarity_dynamics"] > 0.95


def test_too_few_valid_frames_scores_nothing():
    values = _wave(frames=8)
    result = _module().compare_trajectories(_trajectory(values), _trajectory(values))

    assert result["expression_similarity"] is None
    assert result["expression_similarity_distribution"] is None
    assert "too_few_valid_frames" in result["expression_similarity_flags"]


def test_frames_without_a_face_are_dropped_not_zero_filled():
    """Invalid frames must not enter the statistics as zeros."""
    values = _wave(frames=60)
    valid = np.ones(60, dtype=bool)
    valid[10:20] = False
    result = _module().compare_trajectories(_trajectory(values, valid=valid), _trajectory(values))

    assert result["expression_similarity"] > 0.9
    assert result["expression_similarity_generation_face_frames"] == 50
    assert result["expression_similarity_coverage"] == 50 / 60


def test_low_coverage_is_flagged():
    values = _wave(frames=60)
    valid = np.zeros(60, dtype=bool)
    valid[:20] = True
    result = _module().compare_trajectories(_trajectory(values, valid=valid), _trajectory(values))

    assert "low_face_visibility" in result["expression_similarity_flags"]


def test_gaze_channels_can_be_excluded():
    """Gaze follows the shot rather than the person, so it must be droppable."""
    from ayase.modules._blendshape_utils import CANONICAL_BLENDSHAPES, GAZE_BLENDSHAPES

    values = _wave(frames=60)
    gazing = values.copy()
    for index, name in enumerate(CANONICAL_BLENDSHAPES):
        if name in GAZE_BLENDSHAPES:
            gazing[:, index] = 0.05

    with_gaze = _module().compare_trajectories(_trajectory(gazing), _trajectory(values))
    without_gaze = _module(exclude_gaze=True).compare_trajectories(
        _trajectory(gazing), _trajectory(values)
    )

    assert without_gaze["expression_similarity"] > with_gaze["expression_similarity"]


def test_scores_stay_inside_their_bounds():
    high = np.ones((60, 52), dtype=np.float64)
    low = np.zeros((60, 52), dtype=np.float64)
    # Two constant clips at opposite extremes: nothing is active anywhere.
    result = _module().compare_trajectories(_trajectory(low), _trajectory(high))

    assert 0.0 <= result["expression_similarity"] <= 1.0
    assert result["expression_similarity_distribution"] == 0.0


def test_result_is_json_serializable():
    values = _wave()
    result = _module().compare_trajectories(_trajectory(values), _trajectory(values))

    assert json.loads(json.dumps(result)) == result


def test_registry_resolves_without_setup():
    from ayase.pipeline import ModuleRegistry

    ModuleRegistry.discover_modules()
    module_class = ModuleRegistry.get_module("expression_similarity")
    assert module_class is not None
    assert module_class()._ml_available is False


def test_metrics_are_written_onto_the_sample():
    from ayase.models import Sample
    from ayase.modules.expression_similarity import ExpressionSimilarityModule

    values = _wave()
    result = _module().compare_trajectories(_trajectory(values), _trajectory(values))
    sample = Sample(path="clip.mp4", is_video=True)
    ExpressionSimilarityModule._store_result(sample, result)

    assert sample.quality_metrics.expression_similarity == 1.0
    assert sample.quality_metrics.expression_similarity_coverage == 1.0

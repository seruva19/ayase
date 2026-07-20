"""Tests for the identity-invariant expression-following metric."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.modules.conftest import _test_module_basics


def _trajectory(values, valid=None, fps=10.0, frame_indices=None):
    from ayase.modules.expression_following import _Trajectory

    coefficients = np.asarray(values, dtype=np.float32)
    count = len(coefficients)
    valid_array = np.ones(count, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    indices = np.arange(count) if frame_indices is None else np.asarray(frame_indices)
    coefficients = coefficients.copy()
    coefficients[~valid_array] = np.nan
    return _Trajectory(
        timestamps_sec=np.arange(count, dtype=np.float64) / fps,
        coefficients=coefficients,
        valid=valid_array,
        frame_indices=indices.astype(np.int64),
        fps=fps,
        decoded_frames=count,
        face_frames=int(valid_array.sum()),
    )


def test_expression_following_basics():
    from ayase.modules.expression_following import ExpressionFollowingModule

    _test_module_basics(ExpressionFollowingModule, "expression_following")


def test_normalized_l1_bounds_and_identity():
    from ayase.modules.expression_following import BLENDSHAPE_DIM, ExpressionFollowingModule

    zeros = np.zeros(BLENDSHAPE_DIM, dtype=np.float32)
    ones = np.ones(BLENDSHAPE_DIM, dtype=np.float32)
    assert ExpressionFollowingModule._normalized_l1(zeros, zeros) == 0.0
    assert ExpressionFollowingModule._normalized_l1(zeros, ones) == 1.0


def test_category_name_order_is_canonical():
    from ayase.modules.expression_following import (
        CANONICAL_BLENDSHAPES,
        ExpressionFollowingModule,
    )

    categories = [
        SimpleNamespace(category_name=name, score=index / 100.0)
        for index, name in enumerate(reversed(CANONICAL_BLENDSHAPES))
    ]
    vector = ExpressionFollowingModule._categories_to_vector(categories)
    expected = {
        category.category_name: category.score for category in categories
    }
    assert vector is not None
    assert vector.tolist() == pytest.approx(
        [expected[name] for name in CANONICAL_BLENDSHAPES]
    )
    assert ExpressionFollowingModule._categories_to_vector(categories[:-1]) is None
    assert ExpressionFollowingModule._categories_to_vector(categories + categories[:1]) is None


def test_alignment_equal_fps_preserves_frame_index_values():
    from ayase.modules.expression_following import BLENDSHAPE_DIM, ExpressionFollowingModule

    values = np.stack(
        [np.full(BLENDSHAPE_DIM, value, dtype=np.float32) for value in (0.0, 0.5, 1.0)]
    )
    trajectory = _trajectory(values, fps=10.0)
    aligned, valid = ExpressionFollowingModule._resample(
        trajectory, trajectory.timestamps_sec
    )
    assert valid.all()
    assert aligned == pytest.approx(values)


def test_alignment_resamples_to_lower_common_fps():
    from ayase.modules.expression_following import BLENDSHAPE_DIM, ExpressionFollowingModule

    values = np.stack(
        [np.full(BLENDSHAPE_DIM, value, dtype=np.float32) for value in (0.0, 1.0, 0.0)]
    )
    trajectory = _trajectory(values, fps=2.0)
    aligned, valid = ExpressionFollowingModule._resample(
        trajectory, np.asarray([0.25], dtype=np.float64)
    )
    assert valid.tolist() == [True]
    assert aligned[0] == pytest.approx(np.full(BLENDSHAPE_DIM, 0.5))


def test_alignment_does_not_bridge_missing_face_gap():
    from ayase.modules.expression_following import BLENDSHAPE_DIM, ExpressionFollowingModule

    values = np.stack([np.zeros(BLENDSHAPE_DIM)] * 3)
    trajectory = _trajectory(values, valid=[True, False, True], fps=2.0)
    _, valid = ExpressionFollowingModule._resample(
        trajectory, np.asarray([0.25, 0.75], dtype=np.float64)
    )
    assert valid.tolist() == [False, False]


def test_empty_joint_valid_frames_has_no_score():
    from ayase.modules.expression_following import BLENDSHAPE_DIM, ExpressionFollowingModule

    values = np.stack([np.zeros(BLENDSHAPE_DIM)] * 2)
    module = ExpressionFollowingModule()
    result = module._compare_trajectories(
        _trajectory(values, valid=[True, False]),
        _trajectory(values, valid=[False, True]),
    )
    assert result["expression_following"] is None
    assert result["expression_following_distance"] is None
    assert result["expression_following_coverage"] == 0.0
    assert result["expression_following_flags"] == [
        "no_joint_valid_frames",
        "low_face_visibility",
    ]


def test_low_coverage_flag_threshold():
    from ayase.modules.expression_following import BLENDSHAPE_DIM, ExpressionFollowingModule

    values = np.stack([np.zeros(BLENDSHAPE_DIM)] * 2)
    module = ExpressionFollowingModule({"low_coverage_threshold": 0.5})
    at_threshold = module._compare_trajectories(
        _trajectory(values, valid=[True, False]),
        _trajectory(values, valid=[True, True]),
    )
    assert "low_face_visibility" not in at_threshold["expression_following_flags"]
    module.low_coverage_threshold = 0.51
    below_threshold = module._compare_trajectories(
        _trajectory(values, valid=[True, False]),
        _trajectory(values, valid=[True, True]),
    )
    assert "low_face_visibility" in below_threshold["expression_following_flags"]


def test_largest_face_selected_deterministically():
    from ayase.modules.expression_following import ExpressionFollowingModule

    def face(size):
        return [SimpleNamespace(x=0.0, y=0.0), SimpleNamespace(x=size, y=size)]

    result = SimpleNamespace(
        face_landmarks=[face(0.2), face(0.5), face(0.5)],
        face_blendshapes=[[1], [2], [3]],
    )
    assert ExpressionFollowingModule()._select_face_index(result) == 1


def test_result_is_json_serializable():
    from ayase.modules.expression_following import BLENDSHAPE_DIM, ExpressionFollowingModule

    values = np.stack([np.zeros(BLENDSHAPE_DIM)] * 2)
    result = ExpressionFollowingModule()._compare_trajectories(
        _trajectory(values), _trajectory(values)
    )
    encoded = json.dumps(result, allow_nan=False)
    assert "expression_following_per_frame" in encoded


def test_registry_resolves_without_setup():
    from ayase.pipeline import ModuleRegistry

    ModuleRegistry.discover_modules()
    module_class = ModuleRegistry.get_module("expression_following")
    assert module_class is not None
    assert module_class()._ml_available is False

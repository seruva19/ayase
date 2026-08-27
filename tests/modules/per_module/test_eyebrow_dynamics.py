"""Focused tests for THEval eyebrow dynamics."""

from types import SimpleNamespace

import pytest


def test_module_basics():
    from ayase.modules.eyebrow_dynamics import EyebrowDynamicsModule
    from tests.modules.conftest import _test_module_basics
    _test_module_basics(EyebrowDynamicsModule, "eyebrow_dynamics")


def test_eyebrow_dynamics_matches_mean_absolute_transition():
    from ayase.modules.eyebrow_dynamics import eyebrow_dynamics
    assert eyebrow_dynamics([(0.1, 0.2), (0.3, 0.6)]) == pytest.approx(0.3)
    assert eyebrow_dynamics([(0.1, 0.2)]) is None


def test_normalized_brow_distances_are_scale_invariant():
    from ayase.modules.eyebrow_dynamics import (
        LEFT_BROW, LEFT_EYE, RIGHT_BROW, RIGHT_EYE, normalized_brow_distances,
    )
    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(478)]
    for i in LEFT_EYE:
        landmarks[i] = SimpleNamespace(x=0.0, y=0.0)
    for i in RIGHT_EYE:
        landmarks[i] = SimpleNamespace(x=1.0, y=0.0)
    for i in LEFT_BROW:
        landmarks[i] = SimpleNamespace(x=0.0, y=0.25)
    for i in RIGHT_BROW:
        landmarks[i] = SimpleNamespace(x=1.0, y=0.5)
    assert normalized_brow_distances(landmarks) == pytest.approx((0.25, 0.5))


def test_process_sets_metric(video_sample, monkeypatch):
    from ayase.modules.eyebrow_dynamics import EyebrowDynamicsModule
    module = EyebrowDynamicsModule()
    module._ml_available = True
    monkeypatch.setattr(module, "_score_video", lambda _path: 0.01)
    result = module.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.eyebrow_dynamics_score == pytest.approx(0.01)


def test_process_skips_without_backend(video_sample):
    from ayase.modules.eyebrow_dynamics import EyebrowDynamicsModule
    result = EyebrowDynamicsModule().process(video_sample)
    assert result is video_sample and result.quality_metrics is None

"""Focused tests for THEval lip dynamics."""

from types import SimpleNamespace

import numpy as np
import pytest


def test_module_basics():
    from ayase.modules.lip_dynamics import LipDynamicsModule
    from tests.modules.conftest import _test_module_basics

    _test_module_basics(LipDynamicsModule, "lip_dynamics")


def test_lip_shape_vector_has_all_pairwise_distances():
    from ayase.modules.lip_dynamics import LIP_INDICES, lip_shape_vector

    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(478)]
    for offset, index in enumerate(LIP_INDICES):
        landmarks[index] = SimpleNamespace(x=offset / 256.0, y=0.0)
    vector = lip_shape_vector(landmarks)
    assert vector.shape == (len(LIP_INDICES) * (len(LIP_INDICES) - 1) // 2,)
    assert vector[0] == pytest.approx(1.0)


def test_lip_dynamics_matches_theval_population_std():
    from ayase.modules.lip_dynamics import lip_dynamics

    vectors = [np.array([0.0, 2.0]), np.array([2.0, 4.0])]
    assert lip_dynamics(vectors) == pytest.approx(1.0)
    assert lip_dynamics([]) is None


def test_process_sets_metric(video_sample, monkeypatch):
    from ayase.modules.lip_dynamics import LipDynamicsModule

    module = LipDynamicsModule()
    module._ml_available = True
    monkeypatch.setattr(module, "_score_video", lambda _path: 3.25)
    result = module.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.lip_dynamics_score == pytest.approx(3.25)


def test_process_skips_without_backend(video_sample):
    from ayase.modules.lip_dynamics import LipDynamicsModule

    result = LipDynamicsModule().process(video_sample)
    assert result is video_sample
    assert result.quality_metrics is None

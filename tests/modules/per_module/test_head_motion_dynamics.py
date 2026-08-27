"""Focused tests for THEval head-motion dynamics."""

import numpy as np
import pytest


def test_module_basics():
    from ayase.modules.head_motion_dynamics import HeadMotionDynamicsModule
    from tests.modules.conftest import _test_module_basics
    _test_module_basics(HeadMotionDynamicsModule, "head_motion_dynamics")


def test_head_motion_dynamics_matches_theval_equation():
    from ayase.modules.head_motion_dynamics import head_motion_dynamics
    pitch = [0.0, 1.0, 3.0]
    yaw = roll = [0.0, 0.0, 0.0]
    tx = [0.0, 2.0, 4.0]
    ty = [0.0, 0.0, 0.0]
    avg_std = np.std(pitch) / 3.0
    avg_deriv_var = np.var(np.diff(pitch)) / 3.0
    avg_trans_var = np.var(tx) / 2.0
    expected = np.sqrt(avg_std * avg_deriv_var + avg_trans_var)
    assert head_motion_dynamics(pitch, yaw, roll, tx, ty) == pytest.approx(expected)


def test_process_sets_metric(video_sample, monkeypatch):
    from ayase.modules.head_motion_dynamics import HeadMotionDynamicsModule
    module = HeadMotionDynamicsModule()
    module._ml_available = True
    monkeypatch.setattr(module, "_score_video", lambda _path: 2.5)
    result = module.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.head_motion_dynamics_score == pytest.approx(2.5)


def test_process_skips_without_backend(video_sample):
    from ayase.modules.head_motion_dynamics import HeadMotionDynamicsModule
    result = HeadMotionDynamicsModule().process(video_sample)
    assert result is video_sample and result.quality_metrics is None

"""Tests for the time-free movement-manner similarity metric.

Two properties separate this metric from ``pose_driver_fidelity`` and are held by
the tests below: no score may depend on when something happened or on the frame
rate it was recorded at, and a body part that never entered the frame must yield
no score at all rather than a zero that reads as "not reproduced".
"""

import numpy as np

from tests.modules.conftest import _test_module_basics


def _track(speeds_per_second, joints=17, fps=24.0, moments=40, conf=0.9):
    """A synthetic keypoint track where every joint moves at a fixed speed.

    Positions are already in body scales, matching what ``_track`` returns, so a
    constant step per moment produces a constant speed per second.
    """
    step = speeds_per_second / fps
    track = []
    for index in range(moments):
        points = np.full((joints, 2), index * step, dtype=np.float64)
        scores = np.full(joints, conf, dtype=np.float64)
        track.append((index / fps, points, scores))
    return track


def _module(**config):
    from ayase.modules.motion_manner_similarity import MotionMannerSimilarityModule

    return MotionMannerSimilarityModule(config or None)


def test_motion_manner_similarity_basics():
    from ayase.modules.motion_manner_similarity import MotionMannerSimilarityModule

    _test_module_basics(MotionMannerSimilarityModule, "motion_manner_similarity")


def test_identical_distributions_agree_completely():
    from ayase.modules.motion_manner_similarity import _distribution_agreement

    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert _distribution_agreement(values, values) == 1.0


def test_shifted_distribution_agrees_less():
    from ayase.modules.motion_manner_similarity import _distribution_agreement

    calm = np.array([0.1, 0.12, 0.11, 0.13, 0.1])
    lively = np.array([0.9, 1.1, 1.0, 1.2, 0.95])
    assert _distribution_agreement(calm, lively) < _distribution_agreement(calm, calm)


def test_speed_is_per_second_not_per_frame():
    """The same motion filmed at 16 and at 24 fps must read as the same speed.

    This is the reason the metric divides by elapsed time rather than counting
    displacement per frame: without it a clip rendered at a lower frame rate
    would be reported as a calmer person.
    """
    from ayase.modules.motion_manner_similarity import _speed_distribution

    slow_fps = _speed_distribution(_track(2.0, fps=16.0), range(17), min_conf=0.3)
    fast_fps = _speed_distribution(_track(2.0, fps=24.0), range(17), min_conf=0.3)
    assert slow_fps is not None and fast_fps is not None
    assert np.allclose(np.mean(slow_fps), np.mean(fast_fps), rtol=1e-6)


def test_low_confidence_joints_are_skipped():
    from ayase.modules.motion_manner_similarity import _speed_distribution

    track = _track(1.0, conf=0.1)
    assert _speed_distribution(track, range(17), min_conf=0.3) is None


def test_arm_score_absent_when_wrists_never_in_frame(monkeypatch):
    """A head-and-shoulders framing must leave the arm score unset.

    Reporting a number there would be read as "gesticulation not reproduced",
    when in fact nothing about the arms was observed in either clip.
    """
    module = _module()
    module._backend = object()

    def fake_track(self, path):
        return _track(1.0), 1.0, 0.0

    monkeypatch.setattr(type(module), "_track", fake_track)
    result = module._compare("generated.mp4", "reference.mp4")

    assert result is not None
    assert result["motion_manner_arm_coverage"] == 0.0
    assert "motion_manner_arm_agreement" not in result
    assert "motion_manner_speed_agreement" in result


def test_arm_score_present_when_wrists_visible(monkeypatch):
    module = _module()
    module._backend = object()

    def fake_track(self, path):
        return _track(1.0), 1.0, 1.0

    monkeypatch.setattr(type(module), "_track", fake_track)
    result = module._compare("generated.mp4", "reference.mp4")

    assert result is not None
    assert result["motion_manner_arm_agreement"] == 1.0


def test_amplitude_ratio_keeps_direction(monkeypatch):
    """A clip with a wider spread of speeds must report a ratio above one."""
    module = _module()
    module._backend = object()

    def fake_track(self, path):
        if "generated" in path:
            track = _track(1.0)
            for index, (moment, points, scores) in enumerate(track):
                track[index] = (moment, points * (1.0 + 0.5 * index), scores)
            return track, 1.0, 1.0
        return _track(1.0), 1.0, 1.0

    monkeypatch.setattr(type(module), "_track", fake_track)
    result = module._compare("generated.mp4", "reference.mp4")

    assert result is not None
    assert result["motion_manner_amplitude_ratio"] > 1.0


def test_short_tracks_yield_no_score(monkeypatch):
    module = _module()
    module._backend = object()

    def fake_track(self, path):
        return _track(1.0, moments=1), 0.02, 0.0

    monkeypatch.setattr(type(module), "_track", fake_track)
    assert module._compare("generated.mp4", "reference.mp4") is None


def test_process_without_reference_returns_sample_untouched():
    from pathlib import Path

    from ayase.models import Sample

    module = _module()
    module._backend = object()
    sample = Sample(path=Path("clip.mp4"), is_video=True)
    assert module.process(sample) is sample
    assert sample.quality_metrics is None

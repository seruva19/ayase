"""Focused tests for THEval silent-lip stability."""

from types import SimpleNamespace

import pytest


def test_module_basics():
    from ayase.modules.silent_lip_stability import SilentLipStabilityModule
    from tests.modules.conftest import _test_module_basics

    _test_module_basics(SilentLipStabilityModule, "silent_lip_stability")


def test_silence_frame_indices_filters_short_runs():
    from ayase.modules.silent_lip_stability import silence_frame_indices

    # At 10 fps, the 300 ms minimum is three frames. Speech occupies 3..5 and
    # 8..9, leaving a valid 0..2 run and a too-short 6..7 run.
    result = silence_frame_indices(
        [{"start": 0.3, "end": 0.5}, {"start": 0.8, "end": 0.9}],
        fps=10.0,
        total_frames=10,
    )
    assert result == {0, 1, 2}


def test_normalized_lip_opening_matches_theval_formula():
    from ayase.modules.silent_lip_stability import (
        LOWER_LIP_INDICES,
        UPPER_LIP_INDICES,
        normalized_lip_opening,
    )

    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(478)]
    landmarks[33] = SimpleNamespace(x=0.0, y=0.0)
    landmarks[263] = SimpleNamespace(x=1.0, y=0.0)
    for upper, lower in zip(UPPER_LIP_INDICES, LOWER_LIP_INDICES):
        landmarks[upper] = SimpleNamespace(x=0.5, y=0.4)
        landmarks[lower] = SimpleNamespace(x=0.5, y=0.6)
    assert normalized_lip_opening(landmarks) == pytest.approx(0.2)


def test_silent_lip_mad_matches_upstream_aggregation():
    from ayase.modules.silent_lip_stability import silent_lip_mad

    # Mean=0.2; deviations are 0.1, 0, 0.1; median=0.1.
    assert silent_lip_mad([0.1, 0.2, 0.3]) == pytest.approx(0.1)
    assert silent_lip_mad([0.1]) is None


def test_process_sets_metric_without_replacing_sample(video_sample, monkeypatch):
    from ayase.modules.silent_lip_stability import SilentLipStabilityModule

    module = SilentLipStabilityModule()
    module._ml_available = True
    monkeypatch.setattr(module, "_score_video", lambda _path: 0.0125)
    result = module.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.silent_lip_stability == pytest.approx(0.0125)


def test_process_gracefully_skips_without_backend(video_sample):
    from ayase.modules.silent_lip_stability import SilentLipStabilityModule

    result = SilentLipStabilityModule().process(video_sample)
    assert result is video_sample
    assert result.quality_metrics is None

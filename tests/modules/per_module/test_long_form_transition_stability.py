"""Tests for long-form transition stability."""

import numpy as np

from ..conftest import _test_module_basics


def test_long_form_transition_stability_basics():
    from ayase.modules.long_form_transition_stability import LongFormTransitionStabilityModule

    _test_module_basics(LongFormTransitionStabilityModule, "long_form_transition_stability")


def test_clean_boundary_beats_defective_boundary():
    from ayase.modules.long_form_transition_stability import LongFormTransitionStabilityModule

    module = LongFormTransitionStabilityModule()
    clean = [np.full((32, 32), value, np.float32) for value in np.linspace(0.2, 0.7, 12)]
    defective = clean[:4] + [np.zeros((32, 32), np.float32)] * 5 + clean[9:]
    assert module._technical_score(clean) > module._technical_score(defective)


def test_explicit_boundaries_take_precedence():
    from ayase.modules.long_form_transition_stability import LongFormTransitionStabilityModule

    module = LongFormTransitionStabilityModule({"analysis_fps": 4, "boundaries_sec": [1.0]})
    frames = [np.zeros((8, 8), np.float32) for _ in range(10)]
    assert module._boundary_indices(frames) == [4]


def test_video_returns_same_sample(video_sample):
    from ayase.modules.long_form_transition_stability import LongFormTransitionStabilityModule

    result = LongFormTransitionStabilityModule().process(video_sample)
    assert result is video_sample
    assert result.quality_metrics is not None
    assert result.quality_metrics.long_form_transition_stability is not None

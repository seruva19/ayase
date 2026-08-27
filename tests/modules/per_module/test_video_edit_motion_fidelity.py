"""Focused tests for video-edit motion fidelity."""

import numpy as np
import pytest


def test_module_basics():
    from ayase.modules.video_edit_motion_fidelity import VideoEditMotionFidelityModule
    from tests.modules.conftest import _test_module_basics
    _test_module_basics(VideoEditMotionFidelityModule, "video_edit_motion_fidelity")


def test_identical_trajectory_motion_scores_one():
    from ayase.modules.video_edit_motion_fidelity import trajectory_motion_similarity
    tracks = np.asarray([[[0.1, 0.2], [0.2, 0.2], [0.4, 0.2]]], dtype=np.float32)
    assert trajectory_motion_similarity(tracks, tracks) == pytest.approx(1.0)


def test_opposite_velocity_lowers_score():
    from ayase.modules.video_edit_motion_fidelity import trajectory_motion_similarity
    source = np.asarray([[[0.5, 0.5], [0.7, 0.5], [0.9, 0.5]]], dtype=np.float32)
    opposite = np.asarray([[[0.5, 0.5], [0.3, 0.5], [0.1, 0.5]]], dtype=np.float32)
    assert trajectory_motion_similarity(opposite, source) < 0.0


def test_process_sets_metric(video_sample, tmp_path, monkeypatch):
    from ayase.modules.video_edit_motion_fidelity import VideoEditMotionFidelityModule
    module = VideoEditMotionFidelityModule()
    module._ml_available = True
    video_sample.reference_path = tmp_path / "source.mp4"
    monkeypatch.setattr(module, "_score_pair", lambda *_paths: 0.8)
    result = module.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.video_edit_motion_fidelity == pytest.approx(0.8)


def test_process_skips_without_reference(video_sample):
    from ayase.modules.video_edit_motion_fidelity import VideoEditMotionFidelityModule
    module = VideoEditMotionFidelityModule()
    module._ml_available = True
    assert module.process(video_sample).quality_metrics is None

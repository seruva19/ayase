"""Focused tests for THEval mouth quality."""

from types import SimpleNamespace

import pytest


def test_module_basics():
    from ayase.modules.mouth_quality import MouthQualityModule
    from tests.modules.conftest import _test_module_basics
    _test_module_basics(MouthQualityModule, "mouth_quality")


def test_mouth_bbox_matches_padded_clamped_protocol():
    from ayase.modules.lip_dynamics import LIP_INDICES
    from ayase.modules.mouth_quality import mouth_bbox
    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(478)]
    for index in LIP_INDICES:
        landmarks[index] = SimpleNamespace(x=0.5, y=0.5)
    assert mouth_bbox(landmarks, 100, 80, padding=10) == (40, 30, 60, 50)


def test_process_sets_metric(video_sample, monkeypatch):
    from ayase.modules.mouth_quality import MouthQualityModule
    module = MouthQualityModule()
    module._ml_available = True
    monkeypatch.setattr(module, "_score_video", lambda _path: 62.5)
    result = module.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.mouth_quality_score == pytest.approx(62.5)


def test_process_skips_without_backend(video_sample):
    from ayase.modules.mouth_quality import MouthQualityModule
    result = MouthQualityModule().process(video_sample)
    assert result is video_sample and result.quality_metrics is None

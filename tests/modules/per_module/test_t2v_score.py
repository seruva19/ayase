"""Tests for t2v_score module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_t2v_score_basics():
    from ayase.modules.t2v_score import T2VScoreModule
    _test_module_basics(T2VScoreModule, "t2v_score")

def test_t2v_score_video(video_sample):
    from ayase.modules.t2v_score import T2VScoreModule
    video_sample.quality_metrics = QualityMetrics()
    m = T2VScoreModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

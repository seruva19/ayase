"""Tests for compare2score module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_compare2score_basics():
    from ayase.modules.compare2score import Compare2ScoreModule
    _test_module_basics(Compare2ScoreModule, "compare2score")

def test_compare2score_image(image_sample):
    from ayase.modules.compare2score import Compare2ScoreModule
    image_sample.quality_metrics = QualityMetrics()
    m = Compare2ScoreModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_compare2score_video(video_sample):
    from ayase.modules.compare2score import Compare2ScoreModule
    video_sample.quality_metrics = QualityMetrics()
    m = Compare2ScoreModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for inception_score module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_inception_score_basics():
    from ayase.modules.inception_score import InceptionScoreModule
    _test_module_basics(InceptionScoreModule, "inception_score")

def test_inception_score_image(image_sample):
    from ayase.modules.inception_score import InceptionScoreModule
    image_sample.quality_metrics = QualityMetrics()
    m = InceptionScoreModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_inception_score_video(video_sample):
    from ayase.modules.inception_score import InceptionScoreModule
    video_sample.quality_metrics = QualityMetrics()
    m = InceptionScoreModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for commonsense module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_commonsense_basics():
    from ayase.modules.commonsense import CommonsenseModule
    _test_module_basics(CommonsenseModule, "commonsense")

def test_commonsense_image(image_sample):
    from ayase.modules.commonsense import CommonsenseModule
    image_sample.quality_metrics = QualityMetrics()
    m = CommonsenseModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_commonsense_video(video_sample):
    from ayase.modules.commonsense import CommonsenseModule
    video_sample.quality_metrics = QualityMetrics()
    m = CommonsenseModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

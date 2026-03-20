"""Tests for creativity module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_creativity_basics():
    from ayase.modules.creativity import CreativityModule
    _test_module_basics(CreativityModule, "creativity")

def test_creativity_image(image_sample):
    from ayase.modules.creativity import CreativityModule
    image_sample.quality_metrics = QualityMetrics()
    m = CreativityModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_creativity_video(video_sample):
    from ayase.modules.creativity import CreativityModule
    video_sample.quality_metrics = QualityMetrics()
    m = CreativityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for aesthetic module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_aesthetic_basics():
    from ayase.modules.aesthetic import AestheticModule
    _test_module_basics(AestheticModule, "aesthetic")

def test_aesthetic_image(image_sample):
    from ayase.modules.aesthetic import AestheticModule
    image_sample.quality_metrics = QualityMetrics()
    m = AestheticModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_aesthetic_video(video_sample):
    from ayase.modules.aesthetic import AestheticModule
    video_sample.quality_metrics = QualityMetrics()
    m = AestheticModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

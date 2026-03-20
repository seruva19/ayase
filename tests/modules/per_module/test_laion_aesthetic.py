"""Tests for laion_aesthetic module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_laion_aesthetic_basics():
    from ayase.modules.laion_aesthetic import LAIONAestheticModule
    _test_module_basics(LAIONAestheticModule, "laion_aesthetic")

def test_laion_aesthetic_image(image_sample):
    from ayase.modules.laion_aesthetic import LAIONAestheticModule
    image_sample.quality_metrics = QualityMetrics()
    m = LAIONAestheticModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_laion_aesthetic_video(video_sample):
    from ayase.modules.laion_aesthetic import LAIONAestheticModule
    video_sample.quality_metrics = QualityMetrics()
    m = LAIONAestheticModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

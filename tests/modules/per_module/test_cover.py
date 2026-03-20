"""Tests for cover module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_cover_basics():
    from ayase.modules.cover import COVERModule
    _test_module_basics(COVERModule, "cover")

def test_cover_image(image_sample):
    from ayase.modules.cover import COVERModule
    image_sample.quality_metrics = QualityMetrics()
    m = COVERModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_cover_video(video_sample):
    from ayase.modules.cover import COVERModule
    video_sample.quality_metrics = QualityMetrics()
    m = COVERModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

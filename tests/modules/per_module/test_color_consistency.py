"""Tests for color_consistency module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_color_consistency_basics():
    from ayase.modules.color_consistency import ColorConsistencyModule
    _test_module_basics(ColorConsistencyModule, "color_consistency")

def test_color_consistency_image(image_sample):
    from ayase.modules.color_consistency import ColorConsistencyModule
    image_sample.quality_metrics = QualityMetrics()
    m = ColorConsistencyModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_color_consistency_video(video_sample):
    from ayase.modules.color_consistency import ColorConsistencyModule
    video_sample.quality_metrics = QualityMetrics()
    m = ColorConsistencyModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for basic module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_basic_basics():
    from ayase.modules.basic import BasicCompatModule
    _test_module_basics(BasicCompatModule, "basic")

def test_basic_image(image_sample):
    from ayase.modules.basic import BasicCompatModule
    image_sample.quality_metrics = QualityMetrics()
    m = BasicCompatModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_basic_video(video_sample):
    from ayase.modules.basic import BasicCompatModule
    video_sample.quality_metrics = QualityMetrics()
    m = BasicCompatModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

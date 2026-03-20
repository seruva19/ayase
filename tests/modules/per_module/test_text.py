"""Tests for text module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_text_basics():
    from ayase.modules.text import TextCompatModule
    _test_module_basics(TextCompatModule, "text")

def test_text_image(image_sample):
    from ayase.modules.text import TextCompatModule
    image_sample.quality_metrics = QualityMetrics()
    m = TextCompatModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_text_video(video_sample):
    from ayase.modules.text import TextCompatModule
    video_sample.quality_metrics = QualityMetrics()
    m = TextCompatModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

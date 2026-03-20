"""Tests for text_overlay module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_text_overlay_basics():
    from ayase.modules.text_overlay import TextOverlayModule
    _test_module_basics(TextOverlayModule, "text_overlay")

def test_text_overlay_image(image_sample):
    from ayase.modules.text_overlay import TextOverlayModule
    image_sample.quality_metrics = QualityMetrics()
    m = TextOverlayModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_text_overlay_video(video_sample):
    from ayase.modules.text_overlay import TextOverlayModule
    video_sample.quality_metrics = QualityMetrics()
    m = TextOverlayModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

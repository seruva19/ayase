"""Tests for harmful_content module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_harmful_content_basics():
    from ayase.modules.harmful_content import HarmfulContentModule
    _test_module_basics(HarmfulContentModule, "harmful_content")

def test_harmful_content_image(image_sample):
    from ayase.modules.harmful_content import HarmfulContentModule
    image_sample.quality_metrics = QualityMetrics()
    m = HarmfulContentModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_harmful_content_video(video_sample):
    from ayase.modules.harmful_content import HarmfulContentModule
    video_sample.quality_metrics = QualityMetrics()
    m = HarmfulContentModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

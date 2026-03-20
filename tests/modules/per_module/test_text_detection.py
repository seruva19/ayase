"""Tests for text_detection module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_text_detection_basics():
    from ayase.modules.text import TextDetectionModule
    _test_module_basics(TextDetectionModule, "text_detection")

def test_text_detection_image(image_sample):
    from ayase.modules.text import TextDetectionModule
    image_sample.quality_metrics = QualityMetrics()
    m = TextDetectionModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_text_detection_video(video_sample):
    from ayase.modules.text import TextDetectionModule
    video_sample.quality_metrics = QualityMetrics()
    m = TextDetectionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

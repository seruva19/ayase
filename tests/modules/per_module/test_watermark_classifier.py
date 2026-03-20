"""Tests for watermark_classifier module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_watermark_classifier_basics():
    from ayase.modules.watermark_classifier import WatermarkClassificationModule
    _test_module_basics(WatermarkClassificationModule, "watermark_classifier")

def test_watermark_classifier_image(image_sample):
    from ayase.modules.watermark_classifier import WatermarkClassificationModule
    image_sample.quality_metrics = QualityMetrics()
    m = WatermarkClassificationModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_watermark_classifier_video(video_sample):
    from ayase.modules.watermark_classifier import WatermarkClassificationModule
    video_sample.quality_metrics = QualityMetrics()
    m = WatermarkClassificationModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

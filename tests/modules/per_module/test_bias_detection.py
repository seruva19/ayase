"""Tests for bias_detection module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_bias_detection_basics():
    from ayase.modules.bias_detection import BiasDetectionModule
    _test_module_basics(BiasDetectionModule, "bias_detection")

def test_bias_detection_image(image_sample):
    from ayase.modules.bias_detection import BiasDetectionModule
    image_sample.quality_metrics = QualityMetrics()
    m = BiasDetectionModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_bias_detection_video(video_sample):
    from ayase.modules.bias_detection import BiasDetectionModule
    video_sample.quality_metrics = QualityMetrics()
    m = BiasDetectionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

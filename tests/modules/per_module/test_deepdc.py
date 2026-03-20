"""Tests for deepdc module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_deepdc_basics():
    from ayase.modules.deepdc import DeepDCModule
    _test_module_basics(DeepDCModule, "deepdc")

def test_deepdc_image(image_sample):
    from ayase.modules.deepdc import DeepDCModule
    image_sample.quality_metrics = QualityMetrics()
    m = DeepDCModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_deepdc_video(video_sample):
    from ayase.modules.deepdc import DeepDCModule
    video_sample.quality_metrics = QualityMetrics()
    m = DeepDCModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

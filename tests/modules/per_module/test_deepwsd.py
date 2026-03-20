"""Tests for deepwsd module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_deepwsd_basics():
    from ayase.modules.deepwsd import DeepWSDModule
    _test_module_basics(DeepWSDModule, "deepwsd")

def test_deepwsd_image(image_sample):
    from ayase.modules.deepwsd import DeepWSDModule
    image_sample.quality_metrics = QualityMetrics()
    m = DeepWSDModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_deepwsd_video(video_sample):
    from ayase.modules.deepwsd import DeepWSDModule
    video_sample.quality_metrics = QualityMetrics()
    m = DeepWSDModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

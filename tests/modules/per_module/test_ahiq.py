"""Tests for ahiq module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ahiq_basics():
    from ayase.modules.ahiq import AHIQModule
    _test_module_basics(AHIQModule, "ahiq")

def test_ahiq_image(image_sample):
    from ayase.modules.ahiq import AHIQModule
    image_sample.quality_metrics = QualityMetrics()
    m = AHIQModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_ahiq_video(video_sample):
    from ayase.modules.ahiq import AHIQModule
    video_sample.quality_metrics = QualityMetrics()
    m = AHIQModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

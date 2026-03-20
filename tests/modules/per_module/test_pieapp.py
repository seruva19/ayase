"""Tests for pieapp module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_pieapp_basics():
    from ayase.modules.pieapp import PieAPPModule
    _test_module_basics(PieAPPModule, "pieapp")

def test_pieapp_image(image_sample):
    from ayase.modules.pieapp import PieAPPModule
    image_sample.quality_metrics = QualityMetrics()
    m = PieAPPModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_pieapp_video(video_sample):
    from ayase.modules.pieapp import PieAPPModule
    video_sample.quality_metrics = QualityMetrics()
    m = PieAPPModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

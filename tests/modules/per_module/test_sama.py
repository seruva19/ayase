"""Tests for sama module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_sama_basics():
    from ayase.modules.sama import SAMAModule
    _test_module_basics(SAMAModule, "sama")

def test_sama_image(image_sample):
    from ayase.modules.sama import SAMAModule
    image_sample.quality_metrics = QualityMetrics()
    m = SAMAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_sama_video(video_sample):
    from ayase.modules.sama import SAMAModule
    video_sample.quality_metrics = QualityMetrics()
    m = SAMAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for tifa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_tifa_basics():
    from ayase.modules.tifa import TIFAModule
    _test_module_basics(TIFAModule, "tifa")

def test_tifa_image(image_sample):
    from ayase.modules.tifa import TIFAModule
    image_sample.quality_metrics = QualityMetrics()
    m = TIFAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_tifa_video(video_sample):
    from ayase.modules.tifa import TIFAModule
    video_sample.quality_metrics = QualityMetrics()
    m = TIFAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

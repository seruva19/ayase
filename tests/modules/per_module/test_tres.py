"""Tests for tres module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_tres_basics():
    from ayase.modules.tres import TReSModule
    _test_module_basics(TReSModule, "tres")

def test_tres_image(image_sample):
    from ayase.modules.tres import TReSModule
    image_sample.quality_metrics = QualityMetrics()
    m = TReSModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_tres_video(video_sample):
    from ayase.modules.tres import TReSModule
    video_sample.quality_metrics = QualityMetrics()
    m = TReSModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

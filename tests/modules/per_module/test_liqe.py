"""Tests for liqe module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_liqe_basics():
    from ayase.modules.liqe import LIQEModule
    _test_module_basics(LIQEModule, "liqe")

def test_liqe_image(image_sample):
    from ayase.modules.liqe import LIQEModule
    image_sample.quality_metrics = QualityMetrics()
    m = LIQEModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_liqe_video(video_sample):
    from ayase.modules.liqe import LIQEModule
    video_sample.quality_metrics = QualityMetrics()
    m = LIQEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

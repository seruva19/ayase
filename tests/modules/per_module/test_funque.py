"""Tests for funque module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_funque_basics():
    from ayase.modules.funque import FUNQUEModule
    _test_module_basics(FUNQUEModule, "funque")

def test_funque_image(image_sample):
    from ayase.modules.funque import FUNQUEModule
    image_sample.quality_metrics = QualityMetrics()
    m = FUNQUEModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_funque_video(video_sample):
    from ayase.modules.funque import FUNQUEModule
    video_sample.quality_metrics = QualityMetrics()
    m = FUNQUEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for rapique module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_rapique_basics():
    from ayase.modules.rapique import RAPIQUEModule
    _test_module_basics(RAPIQUEModule, "rapique")

def test_rapique_image(image_sample):
    from ayase.modules.rapique import RAPIQUEModule
    image_sample.quality_metrics = QualityMetrics()
    m = RAPIQUEModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_rapique_video(video_sample):
    from ayase.modules.rapique import RAPIQUEModule
    video_sample.quality_metrics = QualityMetrics()
    m = RAPIQUEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

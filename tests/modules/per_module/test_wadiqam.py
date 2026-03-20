"""Tests for wadiqam module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_wadiqam_basics():
    from ayase.modules.wadiqam import WaDIQaMModule
    _test_module_basics(WaDIQaMModule, "wadiqam")

def test_wadiqam_image(image_sample):
    from ayase.modules.wadiqam import WaDIQaMModule
    image_sample.quality_metrics = QualityMetrics()
    m = WaDIQaMModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_wadiqam_video(video_sample):
    from ayase.modules.wadiqam import WaDIQaMModule
    video_sample.quality_metrics = QualityMetrics()
    m = WaDIQaMModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

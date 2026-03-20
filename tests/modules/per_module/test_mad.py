"""Tests for mad module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_mad_basics():
    from ayase.modules.mad_metric import MADModule
    _test_module_basics(MADModule, "mad")

def test_mad_image(image_sample):
    from ayase.modules.mad_metric import MADModule
    image_sample.quality_metrics = QualityMetrics()
    m = MADModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_mad_video(video_sample):
    from ayase.modules.mad_metric import MADModule
    video_sample.quality_metrics = QualityMetrics()
    m = MADModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

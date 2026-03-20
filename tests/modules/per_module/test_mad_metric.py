"""Tests for mad_metric module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_mad_metric_basics():
    from ayase.modules.mad_metric import MADCompatModule
    _test_module_basics(MADCompatModule, "mad_metric")

def test_mad_metric_image(image_sample):
    from ayase.modules.mad_metric import MADCompatModule
    image_sample.quality_metrics = QualityMetrics()
    m = MADCompatModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_mad_metric_video(video_sample):
    from ayase.modules.mad_metric import MADCompatModule
    video_sample.quality_metrics = QualityMetrics()
    m = MADCompatModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

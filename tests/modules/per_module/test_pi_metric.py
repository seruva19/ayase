"""Tests for pi_metric module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_pi_metric_basics():
    from ayase.modules.pi_metric import PICompatModule
    _test_module_basics(PICompatModule, "pi_metric")

def test_pi_metric_image(image_sample):
    from ayase.modules.pi_metric import PICompatModule
    image_sample.quality_metrics = QualityMetrics()
    m = PICompatModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_pi_metric_video(video_sample):
    from ayase.modules.pi_metric import PICompatModule
    video_sample.quality_metrics = QualityMetrics()
    m = PICompatModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

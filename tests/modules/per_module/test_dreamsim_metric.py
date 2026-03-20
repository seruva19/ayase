"""Tests for dreamsim_metric module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dreamsim_metric_basics():
    from ayase.modules.dreamsim_metric import DreamSimCompatModule
    _test_module_basics(DreamSimCompatModule, "dreamsim_metric")

def test_dreamsim_metric_image(image_sample):
    from ayase.modules.dreamsim_metric import DreamSimCompatModule
    image_sample.quality_metrics = QualityMetrics()
    m = DreamSimCompatModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_dreamsim_metric_video(video_sample):
    from ayase.modules.dreamsim_metric import DreamSimCompatModule
    video_sample.quality_metrics = QualityMetrics()
    m = DreamSimCompatModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

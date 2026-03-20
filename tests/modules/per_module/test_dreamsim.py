"""Tests for dreamsim module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dreamsim_basics():
    from ayase.modules.dreamsim_metric import DreamSimModule
    _test_module_basics(DreamSimModule, "dreamsim")

def test_dreamsim_image(image_sample):
    from ayase.modules.dreamsim_metric import DreamSimModule
    image_sample.quality_metrics = QualityMetrics()
    m = DreamSimModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_dreamsim_video(video_sample):
    from ayase.modules.dreamsim_metric import DreamSimModule
    video_sample.quality_metrics = QualityMetrics()
    m = DreamSimModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

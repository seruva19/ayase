"""Tests for pi module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_pi_basics():
    from ayase.modules.pi_metric import PIModule
    _test_module_basics(PIModule, "pi")

def test_pi_image(image_sample):
    from ayase.modules.pi_metric import PIModule
    image_sample.quality_metrics = QualityMetrics()
    m = PIModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_pi_video(video_sample):
    from ayase.modules.pi_metric import PIModule
    video_sample.quality_metrics = QualityMetrics()
    m = PIModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

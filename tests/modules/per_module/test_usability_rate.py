"""Tests for usability_rate module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_usability_rate_basics():
    from ayase.modules.usability_rate import UsabilityRateModule
    _test_module_basics(UsabilityRateModule, "usability_rate")

def test_usability_rate_image(image_sample):
    from ayase.modules.usability_rate import UsabilityRateModule
    image_sample.quality_metrics = QualityMetrics()
    m = UsabilityRateModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_usability_rate_video(video_sample):
    from ayase.modules.usability_rate import UsabilityRateModule
    video_sample.quality_metrics = QualityMetrics()
    m = UsabilityRateModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for serfiq module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_serfiq_basics():
    from ayase.modules.serfiq import SERFIQModule
    _test_module_basics(SERFIQModule, "serfiq")

def test_serfiq_image(image_sample):
    from ayase.modules.serfiq import SERFIQModule
    image_sample.quality_metrics = QualityMetrics()
    m = SERFIQModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_serfiq_video(video_sample):
    from ayase.modules.serfiq import SERFIQModule
    video_sample.quality_metrics = QualityMetrics()
    m = SERFIQModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

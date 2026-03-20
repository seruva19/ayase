"""Tests for ssimc module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ssimc_basics():
    from ayase.modules.ssimc import SSIMCModule
    _test_module_basics(SSIMCModule, "ssimc")

def test_ssimc_image(image_sample):
    from ayase.modules.ssimc import SSIMCModule
    image_sample.quality_metrics = QualityMetrics()
    m = SSIMCModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_ssimc_video(video_sample):
    from ayase.modules.ssimc import SSIMCModule
    video_sample.quality_metrics = QualityMetrics()
    m = SSIMCModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

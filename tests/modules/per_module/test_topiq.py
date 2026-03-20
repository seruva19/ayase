"""Tests for topiq module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_topiq_basics():
    from ayase.modules.topiq import TOPIQModule
    _test_module_basics(TOPIQModule, "topiq")

def test_topiq_image(image_sample):
    from ayase.modules.topiq import TOPIQModule
    image_sample.quality_metrics = QualityMetrics()
    m = TOPIQModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_topiq_video(video_sample):
    from ayase.modules.topiq import TOPIQModule
    video_sample.quality_metrics = QualityMetrics()
    m = TOPIQModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

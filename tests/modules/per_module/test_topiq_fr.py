"""Tests for topiq_fr module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_topiq_fr_basics():
    from ayase.modules.topiq_fr import TOPIQFRModule
    _test_module_basics(TOPIQFRModule, "topiq_fr")

def test_topiq_fr_image(image_sample):
    from ayase.modules.topiq_fr import TOPIQFRModule
    image_sample.quality_metrics = QualityMetrics()
    m = TOPIQFRModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_topiq_fr_video(video_sample):
    from ayase.modules.topiq_fr import TOPIQFRModule
    video_sample.quality_metrics = QualityMetrics()
    m = TOPIQFRModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

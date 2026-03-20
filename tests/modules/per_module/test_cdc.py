"""Tests for cdc module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_cdc_basics():
    from ayase.modules.cdc import CDCModule
    _test_module_basics(CDCModule, "cdc")

def test_cdc_video(video_sample):
    from ayase.modules.cdc import CDCModule
    video_sample.quality_metrics = QualityMetrics()
    m = CDCModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

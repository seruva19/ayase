"""Tests for sqi module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_sqi_basics():
    from ayase.modules.sqi import SQIModule
    _test_module_basics(SQIModule, "sqi")

def test_sqi_video(video_sample):
    from ayase.modules.sqi import SQIModule
    video_sample.quality_metrics = QualityMetrics()
    m = SQIModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for speedqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_speedqa_basics():
    from ayase.modules.speedqa import SpEEDQAModule
    _test_module_basics(SpEEDQAModule, "speedqa")

def test_speedqa_video(video_sample):
    from ayase.modules.speedqa import SpEEDQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = SpEEDQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

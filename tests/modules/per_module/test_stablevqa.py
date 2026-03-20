"""Tests for stablevqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_stablevqa_basics():
    from ayase.modules.stablevqa import StableVQAModule
    _test_module_basics(StableVQAModule, "stablevqa")

def test_stablevqa_video(video_sample):
    from ayase.modules.stablevqa import StableVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = StableVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

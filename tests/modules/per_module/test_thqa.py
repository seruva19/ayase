"""Tests for thqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_thqa_basics():
    from ayase.modules.thqa import THQAModule
    _test_module_basics(THQAModule, "thqa")

def test_thqa_video(video_sample):
    from ayase.modules.thqa import THQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = THQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

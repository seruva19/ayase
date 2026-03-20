"""Tests for c3dvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_c3dvqa_basics():
    from ayase.modules.c3dvqa import C3DVQAModule
    _test_module_basics(C3DVQAModule, "c3dvqa")

def test_c3dvqa_video(video_sample):
    from ayase.modules.c3dvqa import C3DVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = C3DVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for p1203 module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_p1203_basics():
    from ayase.modules.p1203 import P1203Module
    _test_module_basics(P1203Module, "p1203")

def test_p1203_video(video_sample):
    from ayase.modules.p1203 import P1203Module
    video_sample.quality_metrics = QualityMetrics()
    m = P1203Module()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

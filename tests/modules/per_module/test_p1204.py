"""Tests for p1204 module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_p1204_basics():
    from ayase.modules.p1204 import P1204Module
    _test_module_basics(P1204Module, "p1204")

def test_p1204_video(video_sample):
    from ayase.modules.p1204 import P1204Module
    video_sample.quality_metrics = QualityMetrics()
    m = P1204Module()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

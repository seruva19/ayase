"""Tests for crave module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_crave_basics():
    from ayase.modules.crave import CRAVEModule
    _test_module_basics(CRAVEModule, "crave")

def test_crave_video(video_sample):
    from ayase.modules.crave import CRAVEModule
    video_sample.quality_metrics = QualityMetrics()
    m = CRAVEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

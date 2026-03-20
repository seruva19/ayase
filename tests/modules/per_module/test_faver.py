"""Tests for faver module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_faver_basics():
    from ayase.modules.faver import FAVERModule
    _test_module_basics(FAVERModule, "faver")

def test_faver_video(video_sample):
    from ayase.modules.faver import FAVERModule
    video_sample.quality_metrics = QualityMetrics()
    m = FAVERModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

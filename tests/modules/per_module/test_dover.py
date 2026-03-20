"""Tests for dover module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dover_basics():
    from ayase.modules.dover import DOVERModule
    _test_module_basics(DOVERModule, "dover")

def test_dover_video(video_sample):
    from ayase.modules.dover import DOVERModule
    video_sample.quality_metrics = QualityMetrics()
    m = DOVERModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

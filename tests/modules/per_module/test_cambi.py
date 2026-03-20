"""Tests for cambi module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_cambi_basics():
    from ayase.modules.cambi import CAMBIModule
    _test_module_basics(CAMBIModule, "cambi")

def test_cambi_video(video_sample):
    from ayase.modules.cambi import CAMBIModule
    video_sample.quality_metrics = QualityMetrics()
    m = CAMBIModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for flolpips module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_flolpips_basics():
    from ayase.modules.flolpips import FloLPIPSModule
    _test_module_basics(FloLPIPSModule, "flolpips")

def test_flolpips_video(video_sample):
    from ayase.modules.flolpips import FloLPIPSModule
    video_sample.quality_metrics = QualityMetrics()
    m = FloLPIPSModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

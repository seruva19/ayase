"""Tests for ti_si module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ti_si_basics():
    from ayase.modules.ti_si import TISIModule
    _test_module_basics(TISIModule, "ti_si")

def test_ti_si_video(video_sample):
    from ayase.modules.ti_si import TISIModule
    video_sample.quality_metrics = QualityMetrics()
    m = TISIModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

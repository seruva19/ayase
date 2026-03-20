"""Tests for st_lpips module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_st_lpips_basics():
    from ayase.modules.st_lpips import STLPIPSModule
    _test_module_basics(STLPIPSModule, "st_lpips")

def test_st_lpips_video(video_sample):
    from ayase.modules.st_lpips import STLPIPSModule
    video_sample.quality_metrics = QualityMetrics()
    m = STLPIPSModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for stlpips_selfdist module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_st_lpips_basics():
    from ayase.modules.stlpips_selfdist import STLPIPSModule
    _test_module_basics(STLPIPSModule, "stlpips_selfdist")

def test_st_lpips_video(video_sample):
    from ayase.modules.stlpips_selfdist import STLPIPSModule
    video_sample.quality_metrics = QualityMetrics()
    m = STLPIPSModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

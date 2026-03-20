"""Tests for st_greed module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_st_greed_basics():
    from ayase.modules.st_greed import STGREEDModule
    _test_module_basics(STGREEDModule, "st_greed")

def test_st_greed_video(video_sample):
    from ayase.modules.st_greed import STGREEDModule
    video_sample.quality_metrics = QualityMetrics()
    m = STGREEDModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

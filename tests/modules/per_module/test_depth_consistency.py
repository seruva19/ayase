"""Tests for depth_consistency module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_depth_consistency_basics():
    from ayase.modules.depth_consistency import DepthConsistencyModule
    _test_module_basics(DepthConsistencyModule, "depth_consistency")

def test_depth_consistency_video(video_sample):
    from ayase.modules.depth_consistency import DepthConsistencyModule
    video_sample.quality_metrics = QualityMetrics()
    m = DepthConsistencyModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

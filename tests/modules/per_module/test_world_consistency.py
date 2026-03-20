"""Tests for world_consistency module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_world_consistency_basics():
    from ayase.modules.world_consistency import WorldConsistencyModule
    _test_module_basics(WorldConsistencyModule, "world_consistency")

def test_world_consistency_video(video_sample):
    from ayase.modules.world_consistency import WorldConsistencyModule
    video_sample.quality_metrics = QualityMetrics()
    m = WorldConsistencyModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

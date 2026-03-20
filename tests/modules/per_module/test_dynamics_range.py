"""Tests for dynamics_range module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dynamics_range_basics():
    from ayase.modules.dynamics_range import DynamicsRangeModule
    _test_module_basics(DynamicsRangeModule, "dynamics_range")

def test_dynamics_range_video(video_sample):
    from ayase.modules.dynamics_range import DynamicsRangeModule
    video_sample.quality_metrics = QualityMetrics()
    m = DynamicsRangeModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

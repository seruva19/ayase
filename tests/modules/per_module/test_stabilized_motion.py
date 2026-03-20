"""Tests for stabilized_motion module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_stabilized_motion_basics():
    from ayase.modules.stabilized_motion import StabilizedMotionModule
    _test_module_basics(StabilizedMotionModule, "stabilized_motion")

def test_stabilized_motion_video(video_sample):
    from ayase.modules.stabilized_motion import StabilizedMotionModule
    video_sample.quality_metrics = QualityMetrics()
    m = StabilizedMotionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

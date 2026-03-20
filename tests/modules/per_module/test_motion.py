"""Tests for motion module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_motion_basics():
    from ayase.modules.motion import MotionModule
    _test_module_basics(MotionModule, "motion")

def test_motion_video(video_sample):
    from ayase.modules.motion import MotionModule
    video_sample.quality_metrics = QualityMetrics()
    m = MotionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

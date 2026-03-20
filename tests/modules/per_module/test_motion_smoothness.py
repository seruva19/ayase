"""Tests for motion_smoothness module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_motion_smoothness_basics():
    from ayase.modules.motion_smoothness import MotionSmoothnessModule
    _test_module_basics(MotionSmoothnessModule, "motion_smoothness")

def test_motion_smoothness_video(video_sample):
    from ayase.modules.motion_smoothness import MotionSmoothnessModule
    video_sample.quality_metrics = QualityMetrics()
    m = MotionSmoothnessModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

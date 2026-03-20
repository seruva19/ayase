"""Tests for kandinsky_motion module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_kandinsky_motion_basics():
    from ayase.modules.kandinsky_motion import KandinskyMotionModule
    _test_module_basics(KandinskyMotionModule, "kandinsky_motion")

def test_kandinsky_motion_video(video_sample):
    from ayase.modules.kandinsky_motion import KandinskyMotionModule
    video_sample.quality_metrics = QualityMetrics()
    m = KandinskyMotionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

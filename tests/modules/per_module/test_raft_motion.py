"""Tests for raft_motion module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_raft_motion_basics():
    from ayase.modules.raft_motion import RAFTMotionModule
    _test_module_basics(RAFTMotionModule, "raft_motion")

def test_raft_motion_video(video_sample):
    from ayase.modules.raft_motion import RAFTMotionModule
    video_sample.quality_metrics = QualityMetrics()
    m = RAFTMotionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

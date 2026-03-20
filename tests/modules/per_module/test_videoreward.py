"""Tests for videoreward module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_videoreward_basics():
    from ayase.modules.videoreward import VideoRewardModule
    _test_module_basics(VideoRewardModule, "videoreward")

def test_videoreward_video(video_sample):
    from ayase.modules.videoreward import VideoRewardModule
    video_sample.quality_metrics = QualityMetrics()
    m = VideoRewardModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

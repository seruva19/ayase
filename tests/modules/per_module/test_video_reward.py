"""Tests for video_reward module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_video_reward_basics():
    from ayase.modules.video_reward import VideoRewardModule
    _test_module_basics(VideoRewardModule, "video_reward")

def test_video_reward_image(image_sample):
    from ayase.modules.video_reward import VideoRewardModule
    image_sample.quality_metrics = QualityMetrics()
    m = VideoRewardModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_video_reward_video(video_sample):
    from ayase.modules.video_reward import VideoRewardModule
    video_sample.quality_metrics = QualityMetrics()
    m = VideoRewardModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

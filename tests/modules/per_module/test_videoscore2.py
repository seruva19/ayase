"""Tests for videoscore2 module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_videoscore2_basics():
    from ayase.modules.videoscore2 import VideoScore2Module

    _test_module_basics(VideoScore2Module, "videoscore2")


def test_videoscore2_image(image_sample):
    from ayase.modules.videoscore2 import VideoScore2Module

    image_sample.quality_metrics = QualityMetrics()
    module = VideoScore2Module()
    module.on_mount()
    result = module.process(image_sample)
    assert result is image_sample


def test_videoscore2_video(video_sample):
    from ayase.modules.videoscore2 import VideoScore2Module

    video_sample.quality_metrics = QualityMetrics()
    module = VideoScore2Module()
    module.on_mount()
    result = module.process(video_sample)
    assert result is video_sample

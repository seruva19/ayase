"""Tests for videoscore module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_videoscore_basics():
    from ayase.modules.videoscore import VideoScoreModule
    _test_module_basics(VideoScoreModule, "videoscore")

def test_videoscore_image(image_sample):
    from ayase.modules.videoscore import VideoScoreModule
    image_sample.quality_metrics = QualityMetrics()
    m = VideoScoreModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_videoscore_video(video_sample):
    from ayase.modules.videoscore import VideoScoreModule
    video_sample.quality_metrics = QualityMetrics()
    m = VideoScoreModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

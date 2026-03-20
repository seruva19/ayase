"""Tests for video_text_matching module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_video_text_matching_basics():
    from ayase.modules.video_text_matching import VideoTextMatchingModule
    _test_module_basics(VideoTextMatchingModule, "video_text_matching")

def test_video_text_matching_image(image_sample):
    from ayase.modules.video_text_matching import VideoTextMatchingModule
    image_sample.quality_metrics = QualityMetrics()
    m = VideoTextMatchingModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_video_text_matching_video(video_sample):
    from ayase.modules.video_text_matching import VideoTextMatchingModule
    video_sample.quality_metrics = QualityMetrics()
    m = VideoTextMatchingModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

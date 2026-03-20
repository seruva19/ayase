"""Tests for video_memorability module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_video_memorability_basics():
    from ayase.modules.video_memorability import VideoMemorabilityModule
    _test_module_basics(VideoMemorabilityModule, "video_memorability")

def test_video_memorability_image(image_sample):
    from ayase.modules.video_memorability import VideoMemorabilityModule
    image_sample.quality_metrics = QualityMetrics()
    m = VideoMemorabilityModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_video_memorability_video(video_sample):
    from ayase.modules.video_memorability import VideoMemorabilityModule
    video_sample.quality_metrics = QualityMetrics()
    m = VideoMemorabilityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

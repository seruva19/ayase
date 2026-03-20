"""Tests for video_atlas module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_video_atlas_basics():
    from ayase.modules.video_atlas import VideoATLASModule
    _test_module_basics(VideoATLASModule, "video_atlas")

def test_video_atlas_video(video_sample):
    from ayase.modules.video_atlas import VideoATLASModule
    video_sample.quality_metrics = QualityMetrics()
    m = VideoATLASModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for playback_speed module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_playback_speed_basics():
    from ayase.modules.playback_speed import PlaybackSpeedModule
    _test_module_basics(PlaybackSpeedModule, "playback_speed")

def test_playback_speed_video(video_sample):
    from ayase.modules.playback_speed import PlaybackSpeedModule
    video_sample.quality_metrics = QualityMetrics()
    m = PlaybackSpeedModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

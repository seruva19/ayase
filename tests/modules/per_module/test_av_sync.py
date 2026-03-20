"""Tests for av_sync module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_av_sync_basics():
    from ayase.modules.audio_visual_sync import AudioVisualSyncModule
    _test_module_basics(AudioVisualSyncModule, "av_sync")

def test_av_sync_video(video_sample):
    from ayase.modules.audio_visual_sync import AudioVisualSyncModule
    video_sample.quality_metrics = QualityMetrics()
    m = AudioVisualSyncModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

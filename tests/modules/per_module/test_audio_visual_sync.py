"""Tests for audio_visual_sync module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_audio_visual_sync_basics():
    from ayase.modules.audio_visual_sync import AudioVisualSyncCompatModule
    _test_module_basics(AudioVisualSyncCompatModule, "audio_visual_sync")

def test_audio_visual_sync_image(image_sample):
    from ayase.modules.audio_visual_sync import AudioVisualSyncCompatModule
    image_sample.quality_metrics = QualityMetrics()
    m = AudioVisualSyncCompatModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_audio_visual_sync_video(video_sample):
    from ayase.modules.audio_visual_sync import AudioVisualSyncCompatModule
    video_sample.quality_metrics = QualityMetrics()
    m = AudioVisualSyncCompatModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

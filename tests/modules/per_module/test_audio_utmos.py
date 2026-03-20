"""Tests for audio_utmos module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_audio_utmos_basics():
    from ayase.modules.audio_utmos import AudioUTMOSModule
    _test_module_basics(AudioUTMOSModule, "audio_utmos")

def test_audio_utmos_image(image_sample):
    from ayase.modules.audio_utmos import AudioUTMOSModule
    image_sample.quality_metrics = QualityMetrics()
    m = AudioUTMOSModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_audio_utmos_video(video_sample):
    from ayase.modules.audio_utmos import AudioUTMOSModule
    video_sample.quality_metrics = QualityMetrics()
    m = AudioUTMOSModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

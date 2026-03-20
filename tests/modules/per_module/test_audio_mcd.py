"""Tests for audio_mcd module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_audio_mcd_basics():
    from ayase.modules.audio_mcd import AudioMCDModule
    _test_module_basics(AudioMCDModule, "audio_mcd")

def test_audio_mcd_image(image_sample):
    from ayase.modules.audio_mcd import AudioMCDModule
    image_sample.quality_metrics = QualityMetrics()
    m = AudioMCDModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_audio_mcd_video(video_sample):
    from ayase.modules.audio_mcd import AudioMCDModule
    video_sample.quality_metrics = QualityMetrics()
    m = AudioMCDModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

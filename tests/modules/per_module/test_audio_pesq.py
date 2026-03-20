"""Tests for audio_pesq module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_audio_pesq_basics():
    from ayase.modules.audio_pesq import AudioPESQModule
    _test_module_basics(AudioPESQModule, "audio_pesq")

def test_audio_pesq_image(image_sample):
    from ayase.modules.audio_pesq import AudioPESQModule
    image_sample.quality_metrics = QualityMetrics()
    m = AudioPESQModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_audio_pesq_video(video_sample):
    from ayase.modules.audio_pesq import AudioPESQModule
    video_sample.quality_metrics = QualityMetrics()
    m = AudioPESQModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for audio_estoi module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_audio_estoi_basics():
    from ayase.modules.audio_estoi import AudioESTOIModule
    _test_module_basics(AudioESTOIModule, "audio_estoi")

def test_audio_estoi_image(image_sample):
    from ayase.modules.audio_estoi import AudioESTOIModule
    image_sample.quality_metrics = QualityMetrics()
    m = AudioESTOIModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_audio_estoi_video(video_sample):
    from ayase.modules.audio_estoi import AudioESTOIModule
    video_sample.quality_metrics = QualityMetrics()
    m = AudioESTOIModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

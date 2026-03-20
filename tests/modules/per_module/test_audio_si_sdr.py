"""Tests for audio_si_sdr module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_audio_si_sdr_basics():
    from ayase.modules.audio_si_sdr import AudioSISDRModule
    _test_module_basics(AudioSISDRModule, "audio_si_sdr")

def test_audio_si_sdr_image(image_sample):
    from ayase.modules.audio_si_sdr import AudioSISDRModule
    image_sample.quality_metrics = QualityMetrics()
    m = AudioSISDRModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_audio_si_sdr_video(video_sample):
    from ayase.modules.audio_si_sdr import AudioSISDRModule
    video_sample.quality_metrics = QualityMetrics()
    m = AudioSISDRModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

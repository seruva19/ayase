"""Tests for audio_lpdist module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_audio_lpdist_basics():
    from ayase.modules.audio_logmel_dist import AudioLPDistModule
    _test_module_basics(AudioLPDistModule, "audio_logmel_dist")

def test_audio_lpdist_image(image_sample):
    from ayase.modules.audio_logmel_dist import AudioLPDistModule
    image_sample.quality_metrics = QualityMetrics()
    m = AudioLPDistModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_audio_lpdist_video(video_sample):
    from ayase.modules.audio_logmel_dist import AudioLPDistModule
    video_sample.quality_metrics = QualityMetrics()
    m = AudioLPDistModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

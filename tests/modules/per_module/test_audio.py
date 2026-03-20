"""Tests for audio module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_audio_basics():
    from ayase.modules.audio import AudioModule
    _test_module_basics(AudioModule, "audio")

def test_audio_video(video_sample):
    from ayase.modules.audio import AudioModule
    video_sample.quality_metrics = QualityMetrics()
    m = AudioModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

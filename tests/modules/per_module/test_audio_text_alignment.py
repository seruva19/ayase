"""Tests for audio_text_alignment module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_audio_text_alignment_basics():
    from ayase.modules.audio_text_alignment import AudioTextAlignmentModule
    _test_module_basics(AudioTextAlignmentModule, "audio_text_alignment")

def test_audio_text_alignment_image(image_sample):
    from ayase.modules.audio_text_alignment import AudioTextAlignmentModule
    image_sample.quality_metrics = QualityMetrics()
    m = AudioTextAlignmentModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_audio_text_alignment_video(video_sample):
    from ayase.modules.audio_text_alignment import AudioTextAlignmentModule
    video_sample.quality_metrics = QualityMetrics()
    m = AudioTextAlignmentModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

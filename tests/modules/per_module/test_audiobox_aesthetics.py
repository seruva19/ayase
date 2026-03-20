"""Tests for audiobox_aesthetics module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_audiobox_aesthetics_basics():
    from ayase.modules.audiobox_aesthetics import AudioboxAestheticsModule
    _test_module_basics(AudioboxAestheticsModule, "audiobox_aesthetics")

def test_audiobox_aesthetics_image(image_sample):
    from ayase.modules.audiobox_aesthetics import AudioboxAestheticsModule
    image_sample.quality_metrics = QualityMetrics()
    m = AudioboxAestheticsModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_audiobox_aesthetics_video(video_sample):
    from ayase.modules.audiobox_aesthetics import AudioboxAestheticsModule
    video_sample.quality_metrics = QualityMetrics()
    m = AudioboxAestheticsModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

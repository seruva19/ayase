"""Tests for musiq module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_musiq_basics():
    from ayase.modules.musiq import MUSIQModule
    _test_module_basics(MUSIQModule, "musiq")

def test_musiq_image(image_sample):
    from ayase.modules.musiq import MUSIQModule
    image_sample.quality_metrics = QualityMetrics()
    m = MUSIQModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_musiq_video(video_sample):
    from ayase.modules.musiq import MUSIQModule
    video_sample.quality_metrics = QualityMetrics()
    m = MUSIQModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

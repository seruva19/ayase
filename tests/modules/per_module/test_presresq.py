"""Tests for presresq module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_presresq_basics():
    from ayase.modules.presresq import PreResQModule
    _test_module_basics(PreResQModule, "presresq")

def test_presresq_image(image_sample):
    from ayase.modules.presresq import PreResQModule
    image_sample.quality_metrics = QualityMetrics()
    m = PreResQModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_presresq_video(video_sample):
    from ayase.modules.presresq import PreResQModule
    video_sample.quality_metrics = QualityMetrics()
    m = PreResQModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for ugvq module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ugvq_basics():
    from ayase.modules.ugvq import UGVQModule
    _test_module_basics(UGVQModule, "ugvq")

def test_ugvq_image(image_sample):
    from ayase.modules.ugvq import UGVQModule
    image_sample.quality_metrics = QualityMetrics()
    m = UGVQModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_ugvq_video(video_sample):
    from ayase.modules.ugvq import UGVQModule
    video_sample.quality_metrics = QualityMetrics()
    m = UGVQModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

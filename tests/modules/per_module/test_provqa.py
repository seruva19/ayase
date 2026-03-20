"""Tests for provqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_provqa_basics():
    from ayase.modules.provqa import ProVQAModule
    _test_module_basics(ProVQAModule, "provqa")

def test_provqa_image(image_sample):
    from ayase.modules.provqa import ProVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = ProVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_provqa_video(video_sample):
    from ayase.modules.provqa import ProVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = ProVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

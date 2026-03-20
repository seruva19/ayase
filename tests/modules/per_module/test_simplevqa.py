"""Tests for simplevqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_simplevqa_basics():
    from ayase.modules.simplevqa import SimpleVQAModule
    _test_module_basics(SimpleVQAModule, "simplevqa")

def test_simplevqa_image(image_sample):
    from ayase.modules.simplevqa import SimpleVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = SimpleVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_simplevqa_video(video_sample):
    from ayase.modules.simplevqa import SimpleVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = SimpleVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

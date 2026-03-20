"""Tests for clifvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_clifvqa_basics():
    from ayase.modules.clifvqa import CLiFVQAModule
    _test_module_basics(CLiFVQAModule, "clifvqa")

def test_clifvqa_image(image_sample):
    from ayase.modules.clifvqa import CLiFVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = CLiFVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_clifvqa_video(video_sample):
    from ayase.modules.clifvqa import CLiFVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = CLiFVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for aigcvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_aigcvqa_basics():
    from ayase.modules.aigcvqa import AIGCVQAModule
    _test_module_basics(AIGCVQAModule, "aigcvqa")

def test_aigcvqa_image(image_sample):
    from ayase.modules.aigcvqa import AIGCVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = AIGCVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_aigcvqa_video(video_sample):
    from ayase.modules.aigcvqa import AIGCVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = AIGCVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

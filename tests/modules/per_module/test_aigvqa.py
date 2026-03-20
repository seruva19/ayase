"""Tests for aigvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_aigvqa_basics():
    from ayase.modules.aigvqa import AIGVQAModule
    _test_module_basics(AIGVQAModule, "aigvqa")

def test_aigvqa_image(image_sample):
    from ayase.modules.aigvqa import AIGVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = AIGVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_aigvqa_video(video_sample):
    from ayase.modules.aigvqa import AIGVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = AIGVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for maxvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_maxvqa_basics():
    from ayase.modules.maxvqa import MaxVQAModule
    _test_module_basics(MaxVQAModule, "maxvqa")

def test_maxvqa_image(image_sample):
    from ayase.modules.maxvqa import MaxVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = MaxVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_maxvqa_video(video_sample):
    from ayase.modules.maxvqa import MaxVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = MaxVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

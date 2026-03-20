"""Tests for ptmvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ptmvqa_basics():
    from ayase.modules.ptmvqa import PTMVQAModule
    _test_module_basics(PTMVQAModule, "ptmvqa")

def test_ptmvqa_image(image_sample):
    from ayase.modules.ptmvqa import PTMVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = PTMVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_ptmvqa_video(video_sample):
    from ayase.modules.ptmvqa import PTMVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = PTMVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

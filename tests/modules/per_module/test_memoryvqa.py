"""Tests for memoryvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_memoryvqa_basics():
    from ayase.modules.memoryvqa import MemoryVQAModule
    _test_module_basics(MemoryVQAModule, "memoryvqa")

def test_memoryvqa_image(image_sample):
    from ayase.modules.memoryvqa import MemoryVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = MemoryVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_memoryvqa_video(video_sample):
    from ayase.modules.memoryvqa import MemoryVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = MemoryVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

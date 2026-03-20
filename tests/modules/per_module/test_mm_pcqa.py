"""Tests for mm_pcqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_mm_pcqa_basics():
    from ayase.modules.mm_pcqa import MMPCQAModule
    _test_module_basics(MMPCQAModule, "mm_pcqa")

def test_mm_pcqa_image(image_sample):
    from ayase.modules.mm_pcqa import MMPCQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = MMPCQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_mm_pcqa_video(video_sample):
    from ayase.modules.mm_pcqa import MMPCQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = MMPCQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

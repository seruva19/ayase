"""Tests for unified_vqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_unified_vqa_basics():
    from ayase.modules.unified_vqa import UnifiedVQAModule
    _test_module_basics(UnifiedVQAModule, "unified_vqa")

def test_unified_vqa_image(image_sample):
    from ayase.modules.unified_vqa import UnifiedVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = UnifiedVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_unified_vqa_video(video_sample):
    from ayase.modules.unified_vqa import UnifiedVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = UnifiedVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

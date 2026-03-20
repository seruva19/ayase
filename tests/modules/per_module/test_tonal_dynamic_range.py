"""Tests for tonal_dynamic_range module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_tonal_dynamic_range_basics():
    from ayase.modules.tonal_dynamic_range import TonalDynamicRangeModule
    _test_module_basics(TonalDynamicRangeModule, "tonal_dynamic_range")

def test_tonal_dynamic_range_image(image_sample):
    from ayase.modules.tonal_dynamic_range import TonalDynamicRangeModule
    image_sample.quality_metrics = QualityMetrics()
    m = TonalDynamicRangeModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_tonal_dynamic_range_video(video_sample):
    from ayase.modules.tonal_dynamic_range import TonalDynamicRangeModule
    video_sample.quality_metrics = QualityMetrics()
    m = TonalDynamicRangeModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

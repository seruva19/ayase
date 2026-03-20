"""Tests for multiple_objects module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_multiple_objects_basics():
    from ayase.modules.multiple_objects import MultipleObjectsModule
    _test_module_basics(MultipleObjectsModule, "multiple_objects")

def test_multiple_objects_image(image_sample):
    from ayase.modules.multiple_objects import MultipleObjectsModule
    image_sample.quality_metrics = QualityMetrics()
    m = MultipleObjectsModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_multiple_objects_video(video_sample):
    from ayase.modules.multiple_objects import MultipleObjectsModule
    video_sample.quality_metrics = QualityMetrics()
    m = MultipleObjectsModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

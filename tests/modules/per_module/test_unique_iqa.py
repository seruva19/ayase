"""Tests for unique_iqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_unique_iqa_basics():
    from ayase.modules.unique_iqa import UNIQUECompatModule
    _test_module_basics(UNIQUECompatModule, "unique_iqa")

def test_unique_iqa_image(image_sample):
    from ayase.modules.unique_iqa import UNIQUECompatModule
    image_sample.quality_metrics = QualityMetrics()
    m = UNIQUECompatModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_unique_iqa_video(video_sample):
    from ayase.modules.unique_iqa import UNIQUECompatModule
    video_sample.quality_metrics = QualityMetrics()
    m = UNIQUECompatModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

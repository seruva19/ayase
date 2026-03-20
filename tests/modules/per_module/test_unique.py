"""Tests for unique module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_unique_basics():
    from ayase.modules.unique_iqa import UNIQUEModule
    _test_module_basics(UNIQUEModule, "unique")

def test_unique_image(image_sample):
    from ayase.modules.unique_iqa import UNIQUEModule
    image_sample.quality_metrics = QualityMetrics()
    m = UNIQUEModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_unique_video(video_sample):
    from ayase.modules.unique_iqa import UNIQUEModule
    video_sample.quality_metrics = QualityMetrics()
    m = UNIQUEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

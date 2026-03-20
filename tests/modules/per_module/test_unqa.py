"""Tests for unqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_unqa_basics():
    from ayase.modules.unqa import UNQAModule
    _test_module_basics(UNQAModule, "unqa")

def test_unqa_image(image_sample):
    from ayase.modules.unqa import UNQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = UNQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_unqa_video(video_sample):
    from ayase.modules.unqa import UNQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = UNQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

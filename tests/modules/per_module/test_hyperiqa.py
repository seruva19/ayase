"""Tests for hyperiqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_hyperiqa_basics():
    from ayase.modules.hyperiqa import HyperIQAModule
    _test_module_basics(HyperIQAModule, "hyperiqa")

def test_hyperiqa_image(image_sample):
    from ayase.modules.hyperiqa import HyperIQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = HyperIQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_hyperiqa_video(video_sample):
    from ayase.modules.hyperiqa import HyperIQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = HyperIQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for arniqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_arniqa_basics():
    from ayase.modules.arniqa import ARNIQAModule
    _test_module_basics(ARNIQAModule, "arniqa")

def test_arniqa_image(image_sample):
    from ayase.modules.arniqa import ARNIQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = ARNIQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_arniqa_video(video_sample):
    from ayase.modules.arniqa import ARNIQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = ARNIQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

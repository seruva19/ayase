"""Tests for naturalness module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_naturalness_basics():
    from ayase.modules.naturalness import NaturalnessModule
    _test_module_basics(NaturalnessModule, "naturalness")

def test_naturalness_image(image_sample):
    from ayase.modules.naturalness import NaturalnessModule
    image_sample.quality_metrics = QualityMetrics()
    m = NaturalnessModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_naturalness_video(video_sample):
    from ayase.modules.naturalness import NaturalnessModule
    video_sample.quality_metrics = QualityMetrics()
    m = NaturalnessModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

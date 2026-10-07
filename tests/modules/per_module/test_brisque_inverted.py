"""Tests for naturalness module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_naturalness_basics():
    from ayase.modules.brisque_inverted import NaturalnessModule
    _test_module_basics(NaturalnessModule, "brisque_inverted")

def test_naturalness_image(image_sample):
    from ayase.modules.brisque_inverted import NaturalnessModule
    image_sample.quality_metrics = QualityMetrics()
    m = NaturalnessModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_naturalness_video(video_sample):
    from ayase.modules.brisque_inverted import NaturalnessModule
    video_sample.quality_metrics = QualityMetrics()
    m = NaturalnessModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

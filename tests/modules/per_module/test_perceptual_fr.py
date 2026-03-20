"""Tests for perceptual_fr module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_perceptual_fr_basics():
    from ayase.modules.perceptual_fr import PerceptualFRModule
    _test_module_basics(PerceptualFRModule, "perceptual_fr")

def test_perceptual_fr_image(image_sample):
    from ayase.modules.perceptual_fr import PerceptualFRModule
    image_sample.quality_metrics = QualityMetrics()
    m = PerceptualFRModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_perceptual_fr_video(video_sample):
    from ayase.modules.perceptual_fr import PerceptualFRModule
    video_sample.quality_metrics = QualityMetrics()
    m = PerceptualFRModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for background_diversity module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_background_diversity_basics():
    from ayase.modules.background_diversity import BackgroundDiversityModule
    _test_module_basics(BackgroundDiversityModule, "background_diversity")

def test_background_diversity_image(image_sample):
    from ayase.modules.background_diversity import BackgroundDiversityModule
    image_sample.quality_metrics = QualityMetrics()
    m = BackgroundDiversityModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_background_diversity_video(video_sample):
    from ayase.modules.background_diversity import BackgroundDiversityModule
    video_sample.quality_metrics = QualityMetrics()
    m = BackgroundDiversityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

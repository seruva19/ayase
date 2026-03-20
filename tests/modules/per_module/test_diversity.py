"""Tests for diversity module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_diversity_basics():
    from ayase.modules.diversity_selection import DiversitySelectionModule
    _test_module_basics(DiversitySelectionModule, "diversity")

def test_diversity_image(image_sample):
    from ayase.modules.diversity_selection import DiversitySelectionModule
    image_sample.quality_metrics = QualityMetrics()
    m = DiversitySelectionModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_diversity_video(video_sample):
    from ayase.modules.diversity_selection import DiversitySelectionModule
    video_sample.quality_metrics = QualityMetrics()
    m = DiversitySelectionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

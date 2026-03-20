"""Tests for diversity_selection module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_diversity_selection_basics():
    from ayase.modules.diversity_selection import DiversitySelectionCompatModule
    _test_module_basics(DiversitySelectionCompatModule, "diversity_selection")

def test_diversity_selection_image(image_sample):
    from ayase.modules.diversity_selection import DiversitySelectionCompatModule
    image_sample.quality_metrics = QualityMetrics()
    m = DiversitySelectionCompatModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_diversity_selection_video(video_sample):
    from ayase.modules.diversity_selection import DiversitySelectionCompatModule
    video_sample.quality_metrics = QualityMetrics()
    m = DiversitySelectionCompatModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

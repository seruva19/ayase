"""Tests for semantic_selection module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_semantic_selection_basics():
    from ayase.modules.semantic_selection import SemanticSelectionModule
    _test_module_basics(SemanticSelectionModule, "semantic_selection")

def test_semantic_selection_image(image_sample):
    from ayase.modules.semantic_selection import SemanticSelectionModule
    image_sample.quality_metrics = QualityMetrics()
    m = SemanticSelectionModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_semantic_selection_video(video_sample):
    from ayase.modules.semantic_selection import SemanticSelectionModule
    video_sample.quality_metrics = QualityMetrics()
    m = SemanticSelectionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

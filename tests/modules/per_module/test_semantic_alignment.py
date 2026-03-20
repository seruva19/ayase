"""Tests for semantic_alignment module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_semantic_alignment_basics():
    from ayase.modules.semantic_alignment import SemanticAlignmentModule
    _test_module_basics(SemanticAlignmentModule, "semantic_alignment")

def test_semantic_alignment_video(video_sample):
    from ayase.modules.semantic_alignment import SemanticAlignmentModule
    video_sample.quality_metrics = QualityMetrics()
    m = SemanticAlignmentModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

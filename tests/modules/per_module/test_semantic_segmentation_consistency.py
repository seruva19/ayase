"""Tests for semantic_segmentation_consistency module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_semantic_segmentation_consistency_basics():
    from ayase.modules.semantic_segmentation_consistency import SemanticSegmentationConsistencyModule
    _test_module_basics(SemanticSegmentationConsistencyModule, "semantic_segmentation_consistency")

def test_semantic_segmentation_consistency_video(video_sample):
    from ayase.modules.semantic_segmentation_consistency import SemanticSegmentationConsistencyModule
    video_sample.quality_metrics = QualityMetrics()
    m = SemanticSegmentationConsistencyModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

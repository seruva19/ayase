"""Tests for spatial_relationship module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_spatial_relationship_basics():
    from ayase.modules.spatial_relationship import SpatialRelationshipModule
    _test_module_basics(SpatialRelationshipModule, "spatial_relationship")

def test_spatial_relationship_image(image_sample):
    from ayase.modules.spatial_relationship import SpatialRelationshipModule
    image_sample.quality_metrics = QualityMetrics()
    m = SpatialRelationshipModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_spatial_relationship_video(video_sample):
    from ayase.modules.spatial_relationship import SpatialRelationshipModule
    video_sample.quality_metrics = QualityMetrics()
    m = SpatialRelationshipModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

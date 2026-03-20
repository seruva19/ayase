"""Tests for metadata module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_metadata_basics():
    from ayase.modules.metadata import MetadataModule
    _test_module_basics(MetadataModule, "metadata")

def test_metadata_image(image_sample):
    from ayase.modules.metadata import MetadataModule
    image_sample.quality_metrics = QualityMetrics()
    m = MetadataModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_metadata_video(video_sample):
    from ayase.modules.metadata import MetadataModule
    video_sample.quality_metrics = QualityMetrics()
    m = MetadataModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

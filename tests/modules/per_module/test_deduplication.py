"""Tests for deduplication module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_deduplication_basics():
    from ayase.modules.dedup import DeduplicationModule
    _test_module_basics(DeduplicationModule, "deduplication")

def test_deduplication_image(image_sample):
    from ayase.modules.dedup import DeduplicationModule
    image_sample.quality_metrics = QualityMetrics()
    m = DeduplicationModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_deduplication_video(video_sample):
    from ayase.modules.dedup import DeduplicationModule
    video_sample.quality_metrics = QualityMetrics()
    m = DeduplicationModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

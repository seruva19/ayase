"""Tests for resolution_bucketing module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_resolution_bucketing_basics():
    from ayase.modules.resolution_bucketing import ResolutionBucketingModule
    _test_module_basics(ResolutionBucketingModule, "resolution_bucketing")

def test_resolution_bucketing_image(image_sample):
    from ayase.modules.resolution_bucketing import ResolutionBucketingModule
    image_sample.quality_metrics = QualityMetrics()
    m = ResolutionBucketingModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_resolution_bucketing_video(video_sample):
    from ayase.modules.resolution_bucketing import ResolutionBucketingModule
    video_sample.quality_metrics = QualityMetrics()
    m = ResolutionBucketingModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

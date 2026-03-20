"""Tests for imaging_quality module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_imaging_quality_basics():
    from ayase.modules.imaging_quality import ImagingQualityModule
    _test_module_basics(ImagingQualityModule, "imaging_quality")

def test_imaging_quality_image(image_sample):
    from ayase.modules.imaging_quality import ImagingQualityModule
    image_sample.quality_metrics = QualityMetrics()
    m = ImagingQualityModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_imaging_quality_video(video_sample):
    from ayase.modules.imaging_quality import ImagingQualityModule
    video_sample.quality_metrics = QualityMetrics()
    m = ImagingQualityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

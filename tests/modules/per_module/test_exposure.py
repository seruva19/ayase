"""Tests for exposure module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_exposure_basics():
    from ayase.modules.exposure import ExposureModule
    _test_module_basics(ExposureModule, "exposure")

def test_exposure_image(image_sample):
    from ayase.modules.exposure import ExposureModule
    image_sample.quality_metrics = QualityMetrics()
    m = ExposureModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_exposure_video(video_sample):
    from ayase.modules.exposure import ExposureModule
    video_sample.quality_metrics = QualityMetrics()
    m = ExposureModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

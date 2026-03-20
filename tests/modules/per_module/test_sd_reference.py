"""Tests for sd_reference module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_sd_reference_basics():
    from ayase.modules.sd_reference import SDReferenceModule
    _test_module_basics(SDReferenceModule, "sd_reference")

def test_sd_reference_image(image_sample):
    from ayase.modules.sd_reference import SDReferenceModule
    image_sample.quality_metrics = QualityMetrics()
    m = SDReferenceModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_sd_reference_video(video_sample):
    from ayase.modules.sd_reference import SDReferenceModule
    video_sample.quality_metrics = QualityMetrics()
    m = SDReferenceModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

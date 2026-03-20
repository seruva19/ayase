"""Tests for vsfa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vsfa_basics():
    from ayase.modules.vsfa import VSFAModule
    _test_module_basics(VSFAModule, "vsfa")

def test_vsfa_image(image_sample):
    from ayase.modules.vsfa import VSFAModule
    image_sample.quality_metrics = QualityMetrics()
    m = VSFAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_vsfa_video(video_sample):
    from ayase.modules.vsfa import VSFAModule
    video_sample.quality_metrics = QualityMetrics()
    m = VSFAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

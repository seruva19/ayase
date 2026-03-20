"""Tests for vtss module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vtss_basics():
    from ayase.modules.vtss import VTSSModule
    _test_module_basics(VTSSModule, "vtss")

def test_vtss_image(image_sample):
    from ayase.modules.vtss import VTSSModule
    image_sample.quality_metrics = QualityMetrics()
    m = VTSSModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_vtss_video(video_sample):
    from ayase.modules.vtss import VTSSModule
    video_sample.quality_metrics = QualityMetrics()
    m = VTSSModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

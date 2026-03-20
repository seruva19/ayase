"""Tests for cpbd module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_cpbd_basics():
    from ayase.modules.cpbd import CPBDModule
    _test_module_basics(CPBDModule, "cpbd")

def test_cpbd_image(image_sample):
    from ayase.modules.cpbd import CPBDModule
    image_sample.quality_metrics = QualityMetrics()
    m = CPBDModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_cpbd_video(video_sample):
    from ayase.modules.cpbd import CPBDModule
    video_sample.quality_metrics = QualityMetrics()
    m = CPBDModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

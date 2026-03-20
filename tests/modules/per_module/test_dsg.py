"""Tests for dsg module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dsg_basics():
    from ayase.modules.dsg import DSGModule
    _test_module_basics(DSGModule, "dsg")

def test_dsg_image(image_sample):
    from ayase.modules.dsg import DSGModule
    image_sample.quality_metrics = QualityMetrics()
    m = DSGModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_dsg_video(video_sample):
    from ayase.modules.dsg import DSGModule
    video_sample.quality_metrics = QualityMetrics()
    m = DSGModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

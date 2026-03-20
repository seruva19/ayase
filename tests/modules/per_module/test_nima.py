"""Tests for nima module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_nima_basics():
    from ayase.modules.nima import NIMAModule
    _test_module_basics(NIMAModule, "nima")

def test_nima_image(image_sample):
    from ayase.modules.nima import NIMAModule
    image_sample.quality_metrics = QualityMetrics()
    m = NIMAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_nima_video(video_sample):
    from ayase.modules.nima import NIMAModule
    video_sample.quality_metrics = QualityMetrics()
    m = NIMAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

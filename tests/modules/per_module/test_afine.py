"""Tests for afine module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_afine_basics():
    from ayase.modules.afine import AFINEModule
    _test_module_basics(AFINEModule, "afine")

def test_afine_image(image_sample):
    from ayase.modules.afine import AFINEModule
    image_sample.quality_metrics = QualityMetrics()
    m = AFINEModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_afine_video(video_sample):
    from ayase.modules.afine import AFINEModule
    video_sample.quality_metrics = QualityMetrics()
    m = AFINEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

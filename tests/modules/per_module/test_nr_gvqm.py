"""Tests for nr_gvqm module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_nr_gvqm_basics():
    from ayase.modules.nr_gvqm import NRGVQMModule
    _test_module_basics(NRGVQMModule, "nr_gvqm")

def test_nr_gvqm_image(image_sample):
    from ayase.modules.nr_gvqm import NRGVQMModule
    image_sample.quality_metrics = QualityMetrics()
    m = NRGVQMModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_nr_gvqm_video(video_sample):
    from ayase.modules.nr_gvqm import NRGVQMModule
    video_sample.quality_metrics = QualityMetrics()
    m = NRGVQMModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for nrqm module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_nrqm_basics():
    from ayase.modules.nrqm import NRQMModule
    _test_module_basics(NRQMModule, "nrqm")

def test_nrqm_image(image_sample):
    from ayase.modules.nrqm import NRQMModule
    image_sample.quality_metrics = QualityMetrics()
    m = NRQMModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_nrqm_video(video_sample):
    from ayase.modules.nrqm import NRQMModule
    video_sample.quality_metrics = QualityMetrics()
    m = NRQMModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

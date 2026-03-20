"""Tests for hdr_vqm module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_hdr_vqm_basics():
    from ayase.modules.hdr_vqm import HDRVQMModule
    _test_module_basics(HDRVQMModule, "hdr_vqm")

def test_hdr_vqm_image(image_sample):
    from ayase.modules.hdr_vqm import HDRVQMModule
    image_sample.quality_metrics = QualityMetrics()
    m = HDRVQMModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_hdr_vqm_video(video_sample):
    from ayase.modules.hdr_vqm import HDRVQMModule
    video_sample.quality_metrics = QualityMetrics()
    m = HDRVQMModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

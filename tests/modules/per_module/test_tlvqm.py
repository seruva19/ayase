"""Tests for tlvqm module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_tlvqm_basics():
    from ayase.modules.tlvqm import TLVQMModule
    _test_module_basics(TLVQMModule, "tlvqm")

def test_tlvqm_image(image_sample):
    from ayase.modules.tlvqm import TLVQMModule
    image_sample.quality_metrics = QualityMetrics()
    m = TLVQMModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_tlvqm_video(video_sample):
    from ayase.modules.tlvqm import TLVQMModule
    video_sample.quality_metrics = QualityMetrics()
    m = TLVQMModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

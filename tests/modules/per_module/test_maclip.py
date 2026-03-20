"""Tests for maclip module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_maclip_basics():
    from ayase.modules.maclip import MACLIPModule
    _test_module_basics(MACLIPModule, "maclip")

def test_maclip_image(image_sample):
    from ayase.modules.maclip import MACLIPModule
    image_sample.quality_metrics = QualityMetrics()
    m = MACLIPModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_maclip_video(video_sample):
    from ayase.modules.maclip import MACLIPModule
    video_sample.quality_metrics = QualityMetrics()
    m = MACLIPModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

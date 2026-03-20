"""Tests for ckdn module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ckdn_basics():
    from ayase.modules.ckdn import CKDNModule
    _test_module_basics(CKDNModule, "ckdn")

def test_ckdn_image(image_sample):
    from ayase.modules.ckdn import CKDNModule
    image_sample.quality_metrics = QualityMetrics()
    m = CKDNModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_ckdn_video(video_sample):
    from ayase.modules.ckdn import CKDNModule
    video_sample.quality_metrics = QualityMetrics()
    m = CKDNModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

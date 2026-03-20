"""Tests for qclip module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_qclip_basics():
    from ayase.modules.qclip import QCLIPModule
    _test_module_basics(QCLIPModule, "qclip")

def test_qclip_image(image_sample):
    from ayase.modules.qclip import QCLIPModule
    image_sample.quality_metrics = QualityMetrics()
    m = QCLIPModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_qclip_video(video_sample):
    from ayase.modules.qclip import QCLIPModule
    video_sample.quality_metrics = QualityMetrics()
    m = QCLIPModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

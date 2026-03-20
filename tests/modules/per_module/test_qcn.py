"""Tests for qcn module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_qcn_basics():
    from ayase.modules.qcn import QCNModule
    _test_module_basics(QCNModule, "qcn")

def test_qcn_image(image_sample):
    from ayase.modules.qcn import QCNModule
    image_sample.quality_metrics = QualityMetrics()
    m = QCNModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_qcn_video(video_sample):
    from ayase.modules.qcn import QCNModule
    video_sample.quality_metrics = QualityMetrics()
    m = QCNModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

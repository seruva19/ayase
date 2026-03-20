"""Tests for q_align module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_q_align_basics():
    from ayase.modules.q_align import QAlignModule
    _test_module_basics(QAlignModule, "q_align")

def test_q_align_image(image_sample):
    from ayase.modules.q_align import QAlignModule
    image_sample.quality_metrics = QualityMetrics()
    m = QAlignModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_q_align_video(video_sample):
    from ayase.modules.q_align import QAlignModule
    video_sample.quality_metrics = QualityMetrics()
    m = QAlignModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for maniqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_maniqa_basics():
    from ayase.modules.maniqa import MANIQAModule
    _test_module_basics(MANIQAModule, "maniqa")

def test_maniqa_image(image_sample):
    from ayase.modules.maniqa import MANIQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = MANIQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_maniqa_video(video_sample):
    from ayase.modules.maniqa import MANIQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = MANIQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for ilniqe module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ilniqe_basics():
    from ayase.modules.ilniqe import ILNIQEModule
    _test_module_basics(ILNIQEModule, "ilniqe")

def test_ilniqe_image(image_sample):
    from ayase.modules.ilniqe import ILNIQEModule
    image_sample.quality_metrics = QualityMetrics()
    m = ILNIQEModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_ilniqe_video(video_sample):
    from ayase.modules.ilniqe import ILNIQEModule
    video_sample.quality_metrics = QualityMetrics()
    m = ILNIQEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

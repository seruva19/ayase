"""Tests for piqe module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_piqe_basics():
    from ayase.modules.piqe import PIQEModule
    _test_module_basics(PIQEModule, "piqe")

def test_piqe_image(image_sample):
    from ayase.modules.piqe import PIQEModule
    image_sample.quality_metrics = QualityMetrics()
    m = PIQEModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_piqe_video(video_sample):
    from ayase.modules.piqe import PIQEModule
    video_sample.quality_metrics = QualityMetrics()
    m = PIQEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

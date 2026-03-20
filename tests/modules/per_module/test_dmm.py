"""Tests for dmm module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dmm_basics():
    from ayase.modules.dmm import DMMModule
    _test_module_basics(DMMModule, "dmm")

def test_dmm_image(image_sample):
    from ayase.modules.dmm import DMMModule
    image_sample.quality_metrics = QualityMetrics()
    m = DMMModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_dmm_video(video_sample):
    from ayase.modules.dmm import DMMModule
    video_sample.quality_metrics = QualityMetrics()
    m = DMMModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

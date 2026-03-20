"""Tests for sr4kvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_sr4kvqa_basics():
    from ayase.modules.sr4kvqa import SR4KVQAModule
    _test_module_basics(SR4KVQAModule, "sr4kvqa")

def test_sr4kvqa_image(image_sample):
    from ayase.modules.sr4kvqa import SR4KVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = SR4KVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_sr4kvqa_video(video_sample):
    from ayase.modules.sr4kvqa import SR4KVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = SR4KVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

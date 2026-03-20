"""Tests for bvqi module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_bvqi_basics():
    from ayase.modules.bvqi import BVQIModule
    _test_module_basics(BVQIModule, "bvqi")

def test_bvqi_image(image_sample):
    from ayase.modules.bvqi import BVQIModule
    image_sample.quality_metrics = QualityMetrics()
    m = BVQIModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_bvqi_video(video_sample):
    from ayase.modules.bvqi import BVQIModule
    video_sample.quality_metrics = QualityMetrics()
    m = BVQIModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

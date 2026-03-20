"""Tests for depth_anything module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_depth_anything_basics():
    from ayase.modules.depth_anything import DepthAnythingModule
    _test_module_basics(DepthAnythingModule, "depth_anything")

def test_depth_anything_image(image_sample):
    from ayase.modules.depth_anything import DepthAnythingModule
    image_sample.quality_metrics = QualityMetrics()
    m = DepthAnythingModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_depth_anything_video(video_sample):
    from ayase.modules.depth_anything import DepthAnythingModule
    video_sample.quality_metrics = QualityMetrics()
    m = DepthAnythingModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

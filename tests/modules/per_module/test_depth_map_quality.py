"""Tests for depth_map_quality module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_depth_map_quality_basics():
    from ayase.modules.depth_map_quality import DepthMapQualityModule
    _test_module_basics(DepthMapQualityModule, "depth_map_quality")

def test_depth_map_quality_image(image_sample):
    from ayase.modules.depth_map_quality import DepthMapQualityModule
    image_sample.quality_metrics = QualityMetrics()
    m = DepthMapQualityModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_depth_map_quality_video(video_sample):
    from ayase.modules.depth_map_quality import DepthMapQualityModule
    video_sample.quality_metrics = QualityMetrics()
    m = DepthMapQualityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

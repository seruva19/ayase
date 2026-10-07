"""Tests for movie module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_movie_basics():
    from ayase.modules.gabor_flow_vq import MOVIEModule
    _test_module_basics(MOVIEModule, "gabor_flow_vq")

def test_movie_image(image_sample):
    from ayase.modules.gabor_flow_vq import MOVIEModule
    image_sample.quality_metrics = QualityMetrics()
    m = MOVIEModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_movie_video(video_sample):
    from ayase.modules.gabor_flow_vq import MOVIEModule
    video_sample.quality_metrics = QualityMetrics()
    m = MOVIEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

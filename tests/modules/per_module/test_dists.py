"""Tests for dists module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dists_basics():
    from ayase.modules.dists import DISTSModule
    _test_module_basics(DISTSModule, "dists")

def test_dists_image(image_sample):
    from ayase.modules.dists import DISTSModule
    image_sample.quality_metrics = QualityMetrics()
    m = DISTSModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_dists_video(video_sample):
    from ayase.modules.dists import DISTSModule
    video_sample.quality_metrics = QualityMetrics()
    m = DISTSModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

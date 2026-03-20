"""Tests for videval module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_videval_basics():
    from ayase.modules.videval import VIDEVALModule
    _test_module_basics(VIDEVALModule, "videval")

def test_videval_image(image_sample):
    from ayase.modules.videval import VIDEVALModule
    image_sample.quality_metrics = QualityMetrics()
    m = VIDEVALModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_videval_video(video_sample):
    from ayase.modules.videval import VIDEVALModule
    video_sample.quality_metrics = QualityMetrics()
    m = VIDEVALModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

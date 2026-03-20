"""Tests for magface module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_magface_basics():
    from ayase.modules.magface import MagFaceModule
    _test_module_basics(MagFaceModule, "magface")

def test_magface_image(image_sample):
    from ayase.modules.magface import MagFaceModule
    image_sample.quality_metrics = QualityMetrics()
    m = MagFaceModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_magface_video(video_sample):
    from ayase.modules.magface import MagFaceModule
    video_sample.quality_metrics = QualityMetrics()
    m = MagFaceModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

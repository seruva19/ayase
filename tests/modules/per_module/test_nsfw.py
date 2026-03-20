"""Tests for nsfw module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_nsfw_basics():
    from ayase.modules.nsfw import NSFWModule
    _test_module_basics(NSFWModule, "nsfw")

def test_nsfw_image(image_sample):
    from ayase.modules.nsfw import NSFWModule
    image_sample.quality_metrics = QualityMetrics()
    m = NSFWModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_nsfw_video(video_sample):
    from ayase.modules.nsfw import NSFWModule
    video_sample.quality_metrics = QualityMetrics()
    m = NSFWModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

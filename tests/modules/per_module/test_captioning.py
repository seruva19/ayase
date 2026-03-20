"""Tests for captioning module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_captioning_basics():
    from ayase.modules.captioning import CaptioningModule
    _test_module_basics(CaptioningModule, "captioning")

def test_captioning_image(image_sample):
    from ayase.modules.captioning import CaptioningModule
    image_sample.quality_metrics = QualityMetrics()
    m = CaptioningModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_captioning_video(video_sample):
    from ayase.modules.captioning import CaptioningModule
    video_sample.quality_metrics = QualityMetrics()
    m = CaptioningModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for visqol module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_visqol_basics():
    from ayase.modules.visqol import ViSQOLModule
    _test_module_basics(ViSQOLModule, "visqol")

def test_visqol_image(image_sample):
    from ayase.modules.visqol import ViSQOLModule
    image_sample.quality_metrics = QualityMetrics()
    m = ViSQOLModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_visqol_video(video_sample):
    from ayase.modules.visqol import ViSQOLModule
    video_sample.quality_metrics = QualityMetrics()
    m = ViSQOLModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

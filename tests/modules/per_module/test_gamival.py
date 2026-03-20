"""Tests for gamival module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_gamival_basics():
    from ayase.modules.gamival import GAMIVALModule
    _test_module_basics(GAMIVALModule, "gamival")

def test_gamival_image(image_sample):
    from ayase.modules.gamival import GAMIVALModule
    image_sample.quality_metrics = QualityMetrics()
    m = GAMIVALModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_gamival_video(video_sample):
    from ayase.modules.gamival import GAMIVALModule
    video_sample.quality_metrics = QualityMetrics()
    m = GAMIVALModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

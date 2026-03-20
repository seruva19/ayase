"""Tests for paq2piq module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_paq2piq_basics():
    from ayase.modules.paq2piq import PaQ2PiQModule
    _test_module_basics(PaQ2PiQModule, "paq2piq")

def test_paq2piq_image(image_sample):
    from ayase.modules.paq2piq import PaQ2PiQModule
    image_sample.quality_metrics = QualityMetrics()
    m = PaQ2PiQModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_paq2piq_video(video_sample):
    from ayase.modules.paq2piq import PaQ2PiQModule
    video_sample.quality_metrics = QualityMetrics()
    m = PaQ2PiQModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

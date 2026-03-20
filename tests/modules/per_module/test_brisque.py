"""Tests for brisque module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_brisque_basics():
    from ayase.modules.brisque import BRISQUEModule
    _test_module_basics(BRISQUEModule, "brisque")

def test_brisque_image(image_sample):
    from ayase.modules.brisque import BRISQUEModule
    image_sample.quality_metrics = QualityMetrics()
    m = BRISQUEModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_brisque_video(video_sample):
    from ayase.modules.brisque import BRISQUEModule
    video_sample.quality_metrics = QualityMetrics()
    m = BRISQUEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

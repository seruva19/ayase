"""Tests for qualiclip module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_qualiclip_basics():
    from ayase.modules.qualiclip import QualiCLIPModule
    _test_module_basics(QualiCLIPModule, "qualiclip")

def test_qualiclip_image(image_sample):
    from ayase.modules.qualiclip import QualiCLIPModule
    image_sample.quality_metrics = QualityMetrics()
    m = QualiCLIPModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_qualiclip_video(video_sample):
    from ayase.modules.qualiclip import QualiCLIPModule
    video_sample.quality_metrics = QualityMetrics()
    m = QualiCLIPModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

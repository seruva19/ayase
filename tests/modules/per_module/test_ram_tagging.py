"""Tests for ram_tagging module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ram_tagging_basics():
    from ayase.modules.ram_tagging import RAMTaggingModule
    _test_module_basics(RAMTaggingModule, "ram_tagging")

def test_ram_tagging_image(image_sample):
    from ayase.modules.ram_tagging import RAMTaggingModule
    image_sample.quality_metrics = QualityMetrics()
    m = RAMTaggingModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_ram_tagging_video(video_sample):
    from ayase.modules.ram_tagging import RAMTaggingModule
    video_sample.quality_metrics = QualityMetrics()
    m = RAMTaggingModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

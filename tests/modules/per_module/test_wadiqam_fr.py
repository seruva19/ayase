"""Tests for wadiqam_fr module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_wadiqam_fr_basics():
    from ayase.modules.wadiqam_fr import WaDIQaMFRModule
    _test_module_basics(WaDIQaMFRModule, "wadiqam_fr")

def test_wadiqam_fr_image(image_sample):
    from ayase.modules.wadiqam_fr import WaDIQaMFRModule
    image_sample.quality_metrics = QualityMetrics()
    m = WaDIQaMFRModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_wadiqam_fr_video(video_sample):
    from ayase.modules.wadiqam_fr import WaDIQaMFRModule
    video_sample.quality_metrics = QualityMetrics()
    m = WaDIQaMFRModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

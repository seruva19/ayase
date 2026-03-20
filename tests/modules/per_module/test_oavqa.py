"""Tests for oavqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_oavqa_basics():
    from ayase.modules.oavqa import OAVQAModule
    _test_module_basics(OAVQAModule, "oavqa")

def test_oavqa_image(image_sample):
    from ayase.modules.oavqa import OAVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = OAVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_oavqa_video(video_sample):
    from ayase.modules.oavqa import OAVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = OAVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

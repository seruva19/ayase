"""Tests for discovqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_discovqa_basics():
    from ayase.modules.discovqa import DisCoVQAModule
    _test_module_basics(DisCoVQAModule, "discovqa")

def test_discovqa_image(image_sample):
    from ayase.modules.discovqa import DisCoVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = DisCoVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_discovqa_video(video_sample):
    from ayase.modules.discovqa import DisCoVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = DisCoVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

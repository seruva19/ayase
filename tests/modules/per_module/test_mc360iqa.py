"""Tests for mc360iqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_mc360iqa_basics():
    from ayase.modules.mc360iqa import MC360IQAModule
    _test_module_basics(MC360IQAModule, "mc360iqa")

def test_mc360iqa_image(image_sample):
    from ayase.modules.mc360iqa import MC360IQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = MC360IQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_mc360iqa_video(video_sample):
    from ayase.modules.mc360iqa import MC360IQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = MC360IQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

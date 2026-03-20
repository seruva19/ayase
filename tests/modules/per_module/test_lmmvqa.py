"""Tests for lmmvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_lmmvqa_basics():
    from ayase.modules.lmmvqa import LMMVQAModule
    _test_module_basics(LMMVQAModule, "lmmvqa")

def test_lmmvqa_image(image_sample):
    from ayase.modules.lmmvqa import LMMVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = LMMVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_lmmvqa_video(video_sample):
    from ayase.modules.lmmvqa import LMMVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = LMMVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

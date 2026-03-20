"""Tests for rqvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_rqvqa_basics():
    from ayase.modules.rqvqa import RQVQAModule
    _test_module_basics(RQVQAModule, "rqvqa")

def test_rqvqa_image(image_sample):
    from ayase.modules.rqvqa import RQVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = RQVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_rqvqa_video(video_sample):
    from ayase.modules.rqvqa import RQVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = RQVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

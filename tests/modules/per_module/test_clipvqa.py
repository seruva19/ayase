"""Tests for clipvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_clipvqa_basics():
    from ayase.modules.clipvqa import CLIPVQAModule
    _test_module_basics(CLIPVQAModule, "clipvqa")

def test_clipvqa_image(image_sample):
    from ayase.modules.clipvqa import CLIPVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = CLIPVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_clipvqa_video(video_sample):
    from ayase.modules.clipvqa import CLIPVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = CLIPVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

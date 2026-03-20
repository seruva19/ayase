"""Tests for mdvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_mdvqa_basics():
    from ayase.modules.mdvqa import MDVQAModule
    _test_module_basics(MDVQAModule, "mdvqa")

def test_mdvqa_image(image_sample):
    from ayase.modules.mdvqa import MDVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = MDVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_mdvqa_video(video_sample):
    from ayase.modules.mdvqa import MDVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = MDVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

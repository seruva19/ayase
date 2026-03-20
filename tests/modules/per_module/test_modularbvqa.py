"""Tests for modularbvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_modularbvqa_basics():
    from ayase.modules.modularbvqa import ModularBVQAModule
    _test_module_basics(ModularBVQAModule, "modularbvqa")

def test_modularbvqa_image(image_sample):
    from ayase.modules.modularbvqa import ModularBVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = ModularBVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_modularbvqa_video(video_sample):
    from ayase.modules.modularbvqa import ModularBVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = ModularBVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

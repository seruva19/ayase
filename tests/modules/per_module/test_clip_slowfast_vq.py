"""Tests for modularbvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_modularbvqa_basics():
    from ayase.modules.clip_slowfast_vq import ModularBVQAModule
    _test_module_basics(ModularBVQAModule, "clip_slowfast_vq")

def test_modularbvqa_image(image_sample):
    from ayase.modules.clip_slowfast_vq import ModularBVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = ModularBVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_modularbvqa_video(video_sample):
    from ayase.modules.clip_slowfast_vq import ModularBVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = ModularBVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

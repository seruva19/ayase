"""Tests for promptiqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_promptiqa_basics():
    from ayase.modules.promptiqa import PromptIQAModule
    _test_module_basics(PromptIQAModule, "promptiqa")

def test_promptiqa_image(image_sample):
    from ayase.modules.promptiqa import PromptIQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = PromptIQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_promptiqa_video(video_sample):
    from ayase.modules.promptiqa import PromptIQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = PromptIQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

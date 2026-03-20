"""Tests for clip_iqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_clip_iqa_basics():
    from ayase.modules.clip_iqa import CLIPIQAModule
    _test_module_basics(CLIPIQAModule, "clip_iqa")

def test_clip_iqa_image(image_sample):
    from ayase.modules.clip_iqa import CLIPIQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = CLIPIQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_clip_iqa_video(video_sample):
    from ayase.modules.clip_iqa import CLIPIQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = CLIPIQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

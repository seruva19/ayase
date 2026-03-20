"""Tests for fast_vqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_fast_vqa_basics():
    from ayase.modules.fast_vqa import FastVQAModule
    _test_module_basics(FastVQAModule, "fast_vqa")

def test_fast_vqa_video(video_sample):
    from ayase.modules.fast_vqa import FastVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = FastVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

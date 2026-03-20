"""Tests for internvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_internvqa_basics():
    from ayase.modules.internvqa import InternVQAModule
    _test_module_basics(InternVQAModule, "internvqa")

def test_internvqa_video(video_sample):
    from ayase.modules.internvqa import InternVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = InternVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

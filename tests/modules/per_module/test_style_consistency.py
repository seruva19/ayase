"""Tests for style_consistency module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_style_consistency_basics():
    from ayase.modules.style_consistency import StyleConsistencyModule
    _test_module_basics(StyleConsistencyModule, "style_consistency")

def test_style_consistency_video(video_sample):
    from ayase.modules.style_consistency import StyleConsistencyModule
    video_sample.quality_metrics = QualityMetrics()
    m = StyleConsistencyModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

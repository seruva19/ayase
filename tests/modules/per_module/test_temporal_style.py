"""Tests for temporal_style module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_temporal_style_basics():
    from ayase.modules.temporal_style import TemporalStyleModule
    _test_module_basics(TemporalStyleModule, "temporal_style")

def test_temporal_style_video(video_sample):
    from ayase.modules.temporal_style import TemporalStyleModule
    video_sample.quality_metrics = QualityMetrics()
    m = TemporalStyleModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

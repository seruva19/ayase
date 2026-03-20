"""Tests for temporal_flickering module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_temporal_flickering_basics():
    from ayase.modules.temporal_flickering import TemporalFlickeringModule
    _test_module_basics(TemporalFlickeringModule, "temporal_flickering")

def test_temporal_flickering_video(video_sample):
    from ayase.modules.temporal_flickering import TemporalFlickeringModule
    video_sample.quality_metrics = QualityMetrics()
    m = TemporalFlickeringModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

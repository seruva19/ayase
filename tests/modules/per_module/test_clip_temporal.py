"""Tests for clip_temporal module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_clip_temporal_basics():
    from ayase.modules.clip_temporal import CLIPTemporalModule
    _test_module_basics(CLIPTemporalModule, "clip_temporal")

def test_clip_temporal_video(video_sample):
    from ayase.modules.clip_temporal import CLIPTemporalModule
    video_sample.quality_metrics = QualityMetrics()
    m = CLIPTemporalModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

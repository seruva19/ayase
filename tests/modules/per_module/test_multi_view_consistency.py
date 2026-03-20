"""Tests for multi_view_consistency module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_multi_view_consistency_basics():
    from ayase.modules.multi_view_consistency import MultiViewConsistencyModule
    _test_module_basics(MultiViewConsistencyModule, "multi_view_consistency")

def test_multi_view_consistency_video(video_sample):
    from ayase.modules.multi_view_consistency import MultiViewConsistencyModule
    video_sample.quality_metrics = QualityMetrics()
    m = MultiViewConsistencyModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

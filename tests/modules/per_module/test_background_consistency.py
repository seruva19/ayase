"""Tests for background_consistency module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_background_consistency_basics():
    from ayase.modules.background_consistency import BackgroundConsistencyModule
    _test_module_basics(BackgroundConsistencyModule, "background_consistency")

def test_background_consistency_video(video_sample):
    from ayase.modules.background_consistency import BackgroundConsistencyModule
    video_sample.quality_metrics = QualityMetrics()
    m = BackgroundConsistencyModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

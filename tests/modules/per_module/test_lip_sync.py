"""Tests for lip_sync module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_lip_sync_basics():
    from ayase.modules.lip_sync import LipSyncModule
    _test_module_basics(LipSyncModule, "lip_sync")

def test_lip_sync_video(video_sample):
    from ayase.modules.lip_sync import LipSyncModule
    video_sample.quality_metrics = QualityMetrics()
    m = LipSyncModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

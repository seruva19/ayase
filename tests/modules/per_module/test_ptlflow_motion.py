"""Tests for ptlflow_motion module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ptlflow_motion_basics():
    from ayase.modules.ptlflow_motion import PtlflowMotionModule
    _test_module_basics(PtlflowMotionModule, "ptlflow_motion")

def test_ptlflow_motion_video(video_sample):
    from ayase.modules.ptlflow_motion import PtlflowMotionModule
    video_sample.quality_metrics = QualityMetrics()
    m = PtlflowMotionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

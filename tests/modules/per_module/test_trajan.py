"""Tests for trajan module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_trajan_basics():
    from ayase.modules.trajan import TRAJANModule
    _test_module_basics(TRAJANModule, "trajan")

def test_trajan_video(video_sample):
    from ayase.modules.trajan import TRAJANModule
    video_sample.quality_metrics = QualityMetrics()
    m = TRAJANModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

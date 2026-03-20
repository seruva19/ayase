"""Tests for judder_stutter module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_judder_stutter_basics():
    from ayase.modules.judder_stutter import JudderStutterModule
    _test_module_basics(JudderStutterModule, "judder_stutter")

def test_judder_stutter_video(video_sample):
    from ayase.modules.judder_stutter import JudderStutterModule
    video_sample.quality_metrics = QualityMetrics()
    m = JudderStutterModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

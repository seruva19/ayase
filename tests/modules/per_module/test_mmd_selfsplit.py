"""Tests for mmd_selfsplit module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_jedi_basics():
    from ayase.modules.mmd_selfsplit import JEDiModule
    _test_module_basics(JEDiModule, "mmd_selfsplit")

def test_jedi_video(video_sample):
    from ayase.modules.mmd_selfsplit import JEDiModule
    video_sample.quality_metrics = QualityMetrics()
    m = JEDiModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

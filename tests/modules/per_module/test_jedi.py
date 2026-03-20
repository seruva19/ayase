"""Tests for jedi module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_jedi_basics():
    from ayase.modules.jedi_metric import JEDiModule
    _test_module_basics(JEDiModule, "jedi")

def test_jedi_video(video_sample):
    from ayase.modules.jedi_metric import JEDiModule
    video_sample.quality_metrics = QualityMetrics()
    m = JEDiModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

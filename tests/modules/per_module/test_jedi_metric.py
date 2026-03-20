"""Tests for jedi_metric module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_jedi_metric_basics():
    from ayase.modules.jedi_metric import JEDiCompatModule
    _test_module_basics(JEDiCompatModule, "jedi_metric")

def test_jedi_metric_image(image_sample):
    from ayase.modules.jedi_metric import JEDiCompatModule
    image_sample.quality_metrics = QualityMetrics()
    m = JEDiCompatModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_jedi_metric_video(video_sample):
    from ayase.modules.jedi_metric import JEDiCompatModule
    video_sample.quality_metrics = QualityMetrics()
    m = JEDiCompatModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

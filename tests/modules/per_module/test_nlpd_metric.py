"""Tests for nlpd_metric module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_nlpd_metric_basics():
    from ayase.modules.nlpd_metric import NLPDCompatModule
    _test_module_basics(NLPDCompatModule, "nlpd_metric")

def test_nlpd_metric_image(image_sample):
    from ayase.modules.nlpd_metric import NLPDCompatModule
    image_sample.quality_metrics = QualityMetrics()
    m = NLPDCompatModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_nlpd_metric_video(video_sample):
    from ayase.modules.nlpd_metric import NLPDCompatModule
    video_sample.quality_metrics = QualityMetrics()
    m = NLPDCompatModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

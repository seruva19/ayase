"""Tests for nlpd module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_nlpd_basics():
    from ayase.modules.nlpd_metric import NLPDModule
    _test_module_basics(NLPDModule, "nlpd")

def test_nlpd_image(image_sample):
    from ayase.modules.nlpd_metric import NLPDModule
    image_sample.quality_metrics = QualityMetrics()
    m = NLPDModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_nlpd_video(video_sample):
    from ayase.modules.nlpd_metric import NLPDModule
    video_sample.quality_metrics = QualityMetrics()
    m = NLPDModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

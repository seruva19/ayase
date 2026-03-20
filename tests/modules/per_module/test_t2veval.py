"""Tests for t2veval module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_t2veval_basics():
    from ayase.modules.t2veval import T2VEvalModule
    _test_module_basics(T2VEvalModule, "t2veval")

def test_t2veval_image(image_sample):
    from ayase.modules.t2veval import T2VEvalModule
    image_sample.quality_metrics = QualityMetrics()
    m = T2VEvalModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_t2veval_video(video_sample):
    from ayase.modules.t2veval import T2VEvalModule
    video_sample.quality_metrics = QualityMetrics()
    m = T2VEvalModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

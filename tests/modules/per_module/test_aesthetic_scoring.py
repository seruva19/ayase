"""Tests for aesthetic_scoring module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_aesthetic_scoring_basics():
    from ayase.modules.aesthetic_scoring import AestheticScoringModule
    _test_module_basics(AestheticScoringModule, "aesthetic_scoring")

def test_aesthetic_scoring_image(image_sample):
    from ayase.modules.aesthetic_scoring import AestheticScoringModule
    image_sample.quality_metrics = QualityMetrics()
    m = AestheticScoringModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_aesthetic_scoring_video(video_sample):
    from ayase.modules.aesthetic_scoring import AestheticScoringModule
    video_sample.quality_metrics = QualityMetrics()
    m = AestheticScoringModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for umtscore module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_umtscore_basics():
    from ayase.modules.umtscore import UMTScoreModule
    _test_module_basics(UMTScoreModule, "umtscore")

def test_umtscore_image(image_sample):
    from ayase.modules.umtscore import UMTScoreModule
    image_sample.quality_metrics = QualityMetrics()
    m = UMTScoreModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_umtscore_video(video_sample):
    from ayase.modules.umtscore import UMTScoreModule
    video_sample.quality_metrics = QualityMetrics()
    m = UMTScoreModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

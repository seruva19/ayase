"""Tests for dedup module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dedup_basics():
    from ayase.modules.dedup import DedupCompatModule
    _test_module_basics(DedupCompatModule, "dedup")

def test_dedup_image(image_sample):
    from ayase.modules.dedup import DedupCompatModule
    image_sample.quality_metrics = QualityMetrics()
    m = DedupCompatModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_dedup_video(video_sample):
    from ayase.modules.dedup import DedupCompatModule
    video_sample.quality_metrics = QualityMetrics()
    m = DedupCompatModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

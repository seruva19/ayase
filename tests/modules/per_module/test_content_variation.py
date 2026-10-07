"""Tests for content_variation module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dynamics_range_basics():
    from ayase.modules.content_variation import ContentVariationModule
    _test_module_basics(ContentVariationModule, "content_variation")

def test_dynamics_range_video(video_sample):
    from ayase.modules.content_variation import ContentVariationModule
    video_sample.quality_metrics = QualityMetrics()
    m = ContentVariationModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

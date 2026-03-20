"""Tests for stereoscopic_quality module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_stereoscopic_quality_basics():
    from ayase.modules.stereoscopic_quality import StereoscopicQualityModule
    _test_module_basics(StereoscopicQualityModule, "stereoscopic_quality")

def test_stereoscopic_quality_video(video_sample):
    from ayase.modules.stereoscopic_quality import StereoscopicQualityModule
    video_sample.quality_metrics = QualityMetrics()
    m = StereoscopicQualityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for viideo module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_viideo_basics():
    from ayase.modules.viideo import VIIDEOModule
    _test_module_basics(VIIDEOModule, "viideo")

def test_viideo_video(video_sample):
    from ayase.modules.viideo import VIIDEOModule
    video_sample.quality_metrics = QualityMetrics()
    m = VIIDEOModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

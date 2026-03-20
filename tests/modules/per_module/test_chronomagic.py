"""Tests for chronomagic module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_chronomagic_basics():
    from ayase.modules.chronomagic import ChronoMagicModule
    _test_module_basics(ChronoMagicModule, "chronomagic")

def test_chronomagic_video(video_sample):
    from ayase.modules.chronomagic import ChronoMagicModule
    video_sample.quality_metrics = QualityMetrics()
    m = ChronoMagicModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

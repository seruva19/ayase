"""Tests for spectral_complexity module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_spectral_complexity_basics():
    from ayase.modules.spectral import SpectralComplexityModule
    _test_module_basics(SpectralComplexityModule, "spectral_complexity")

def test_spectral_complexity_video(video_sample):
    from ayase.modules.spectral import SpectralComplexityModule
    video_sample.quality_metrics = QualityMetrics()
    m = SpectralComplexityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

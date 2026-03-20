"""Tests for spectral module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_spectral_basics():
    from ayase.modules.spectral import SpectralCompatModule
    _test_module_basics(SpectralCompatModule, "spectral")

def test_spectral_image(image_sample):
    from ayase.modules.spectral import SpectralCompatModule
    image_sample.quality_metrics = QualityMetrics()
    m = SpectralCompatModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_spectral_video(video_sample):
    from ayase.modules.spectral import SpectralCompatModule
    video_sample.quality_metrics = QualityMetrics()
    m = SpectralCompatModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

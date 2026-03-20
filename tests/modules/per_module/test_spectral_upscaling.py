"""Tests for spectral_upscaling module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_spectral_upscaling_basics():
    from ayase.modules.spectral_upscaling import SpectralUpscalingModule
    _test_module_basics(SpectralUpscalingModule, "spectral_upscaling")

def test_spectral_upscaling_image(image_sample):
    from ayase.modules.spectral_upscaling import SpectralUpscalingModule
    image_sample.quality_metrics = QualityMetrics()
    m = SpectralUpscalingModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_spectral_upscaling_video(video_sample):
    from ayase.modules.spectral_upscaling import SpectralUpscalingModule
    video_sample.quality_metrics = QualityMetrics()
    m = SpectralUpscalingModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for grafiqs module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_grafiqs_basics():
    from ayase.modules.grafiqs import GraFIQsModule
    _test_module_basics(GraFIQsModule, "grafiqs")

def test_grafiqs_image(image_sample):
    from ayase.modules.grafiqs import GraFIQsModule
    image_sample.quality_metrics = QualityMetrics()
    m = GraFIQsModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_grafiqs_video(video_sample):
    from ayase.modules.grafiqs import GraFIQsModule
    video_sample.quality_metrics = QualityMetrics()
    m = GraFIQsModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

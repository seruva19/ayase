"""Tests for production_quality module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_production_quality_basics():
    from ayase.modules.production_quality import ProductionQualityModule
    _test_module_basics(ProductionQualityModule, "production_quality")

def test_production_quality_image(image_sample):
    from ayase.modules.production_quality import ProductionQualityModule
    image_sample.quality_metrics = QualityMetrics()
    m = ProductionQualityModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_production_quality_video(video_sample):
    from ayase.modules.production_quality import ProductionQualityModule
    video_sample.quality_metrics = QualityMetrics()
    m = ProductionQualityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

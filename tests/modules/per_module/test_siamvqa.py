"""Tests for siamvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_siamvqa_basics():
    from ayase.modules.siamvqa import SiamVQAModule
    _test_module_basics(SiamVQAModule, "siamvqa")

def test_siamvqa_image(image_sample):
    from ayase.modules.siamvqa import SiamVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = SiamVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_siamvqa_video(video_sample):
    from ayase.modules.siamvqa import SiamVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = SiamVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

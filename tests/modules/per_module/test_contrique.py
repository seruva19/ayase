"""Tests for contrique module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_contrique_basics():
    from ayase.modules.contrique import CONTRIQUEModule
    _test_module_basics(CONTRIQUEModule, "contrique")

def test_contrique_image(image_sample):
    from ayase.modules.contrique import CONTRIQUEModule
    image_sample.quality_metrics = QualityMetrics()
    m = CONTRIQUEModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_contrique_video(video_sample):
    from ayase.modules.contrique import CONTRIQUEModule
    video_sample.quality_metrics = QualityMetrics()
    m = CONTRIQUEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

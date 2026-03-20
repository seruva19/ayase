"""Tests for crfiqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_crfiqa_basics():
    from ayase.modules.crfiqa import CRFIQAModule
    _test_module_basics(CRFIQAModule, "crfiqa")

def test_crfiqa_image(image_sample):
    from ayase.modules.crfiqa import CRFIQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = CRFIQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_crfiqa_video(video_sample):
    from ayase.modules.crfiqa import CRFIQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = CRFIQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

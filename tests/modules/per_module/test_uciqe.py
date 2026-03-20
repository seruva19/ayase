"""Tests for uciqe module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_uciqe_basics():
    from ayase.modules.uciqe import UCIQEModule
    _test_module_basics(UCIQEModule, "uciqe")

def test_uciqe_image(image_sample):
    from ayase.modules.uciqe import UCIQEModule
    image_sample.quality_metrics = QualityMetrics()
    m = UCIQEModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_uciqe_video(video_sample):
    from ayase.modules.uciqe import UCIQEModule
    video_sample.quality_metrics = QualityMetrics()
    m = UCIQEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for cnniqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_cnniqa_basics():
    from ayase.modules.cnniqa import CNNIQAModule
    _test_module_basics(CNNIQAModule, "cnniqa")

def test_cnniqa_image(image_sample):
    from ayase.modules.cnniqa import CNNIQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = CNNIQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_cnniqa_video(video_sample):
    from ayase.modules.cnniqa import CNNIQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = CNNIQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

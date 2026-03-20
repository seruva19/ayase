"""Tests for vqa2 module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vqa2_basics():
    from ayase.modules.vqa2 import VQA2Module
    _test_module_basics(VQA2Module, "vqa2")

def test_vqa2_image(image_sample):
    from ayase.modules.vqa2 import VQA2Module
    image_sample.quality_metrics = QualityMetrics()
    m = VQA2Module()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_vqa2_video(video_sample):
    from ayase.modules.vqa2 import VQA2Module
    video_sample.quality_metrics = QualityMetrics()
    m = VQA2Module()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

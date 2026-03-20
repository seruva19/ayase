"""Tests for 4k_vqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_4k_vqa_basics():
    from ayase.modules.hdr_sdr_vqa import FourKVQAModule
    _test_module_basics(FourKVQAModule, "4k_vqa")

def test_4k_vqa_video(video_sample):
    from ayase.modules.hdr_sdr_vqa import FourKVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = FourKVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

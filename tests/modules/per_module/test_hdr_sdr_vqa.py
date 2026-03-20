"""Tests for hdr_sdr_vqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_hdr_sdr_vqa_basics():
    from ayase.modules.hdr_sdr_vqa import HDRSDRVQAModule
    _test_module_basics(HDRSDRVQAModule, "hdr_sdr_vqa")

def test_hdr_sdr_vqa_video(video_sample):
    from ayase.modules.hdr_sdr_vqa import HDRSDRVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = HDRSDRVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

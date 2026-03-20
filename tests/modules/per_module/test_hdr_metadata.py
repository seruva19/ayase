"""Tests for hdr_metadata module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_hdr_metadata_basics():
    from ayase.modules.hdr_metadata import HDRMetadataModule
    _test_module_basics(HDRMetadataModule, "hdr_metadata")

def test_hdr_metadata_video(video_sample):
    from ayase.modules.hdr_metadata import HDRMetadataModule
    video_sample.quality_metrics = QualityMetrics()
    m = HDRMetadataModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

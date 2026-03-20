"""Tests for codec_specific_quality module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_codec_specific_quality_basics():
    from ayase.modules.codec_specific_quality import CodecSpecificQualityModule
    _test_module_basics(CodecSpecificQualityModule, "codec_specific_quality")

def test_codec_specific_quality_video(video_sample):
    from ayase.modules.codec_specific_quality import CodecSpecificQualityModule
    video_sample.quality_metrics = QualityMetrics()
    m = CodecSpecificQualityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

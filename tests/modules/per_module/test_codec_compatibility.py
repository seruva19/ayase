"""Tests for codec_compatibility module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_codec_compatibility_basics():
    from ayase.modules.codec_compatibility import CodecCompatibilityModule
    _test_module_basics(CodecCompatibilityModule, "codec_compatibility")

def test_codec_compatibility_video(video_sample):
    from ayase.modules.codec_compatibility import CodecCompatibilityModule
    video_sample.quality_metrics = QualityMetrics()
    m = CodecCompatibilityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

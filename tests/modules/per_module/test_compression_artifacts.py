"""Tests for compression_artifacts module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_compression_artifacts_basics():
    from ayase.modules.compression_artifacts import CompressionArtifactsModule
    _test_module_basics(CompressionArtifactsModule, "compression_artifacts")

def test_compression_artifacts_video(video_sample):
    from ayase.modules.compression_artifacts import CompressionArtifactsModule
    video_sample.quality_metrics = QualityMetrics()
    m = CompressionArtifactsModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

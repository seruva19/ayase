"""Tests for embedding module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_embedding_basics():
    from ayase.modules.embedding import EmbeddingModule
    _test_module_basics(EmbeddingModule, "embedding")

def test_embedding_image(image_sample):
    from ayase.modules.embedding import EmbeddingModule
    image_sample.quality_metrics = QualityMetrics()
    m = EmbeddingModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_embedding_video(video_sample):
    from ayase.modules.embedding import EmbeddingModule
    video_sample.quality_metrics = QualityMetrics()
    m = EmbeddingModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

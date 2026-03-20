"""Tests for knowledge_graph module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_knowledge_graph_basics():
    from ayase.modules.knowledge_graph import KnowledgeGraphModule
    _test_module_basics(KnowledgeGraphModule, "knowledge_graph")

def test_knowledge_graph_image(image_sample):
    from ayase.modules.knowledge_graph import KnowledgeGraphModule
    image_sample.quality_metrics = QualityMetrics()
    m = KnowledgeGraphModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_knowledge_graph_video(video_sample):
    from ayase.modules.knowledge_graph import KnowledgeGraphModule
    video_sample.quality_metrics = QualityMetrics()
    m = KnowledgeGraphModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

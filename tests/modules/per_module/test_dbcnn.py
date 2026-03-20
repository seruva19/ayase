"""Tests for dbcnn module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dbcnn_basics():
    from ayase.modules.dbcnn import DBCNNModule
    _test_module_basics(DBCNNModule, "dbcnn")

def test_dbcnn_image(image_sample):
    from ayase.modules.dbcnn import DBCNNModule
    image_sample.quality_metrics = QualityMetrics()
    m = DBCNNModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_dbcnn_video(video_sample):
    from ayase.modules.dbcnn import DBCNNModule
    video_sample.quality_metrics = QualityMetrics()
    m = DBCNNModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

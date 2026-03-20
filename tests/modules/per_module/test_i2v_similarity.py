"""Tests for i2v_similarity module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_i2v_similarity_basics():
    from ayase.modules.i2v_similarity import I2VSimilarityModule
    _test_module_basics(I2VSimilarityModule, "i2v_similarity")

def test_i2v_similarity_video(video_sample):
    from ayase.modules.i2v_similarity import I2VSimilarityModule
    video_sample.quality_metrics = QualityMetrics()
    m = I2VSimilarityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

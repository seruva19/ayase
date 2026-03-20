"""Tests for aigv_assessor module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_aigv_assessor_basics():
    from ayase.modules.aigv_assessor import AIGVAssessorModule
    _test_module_basics(AIGVAssessorModule, "aigv_assessor")

def test_aigv_assessor_video(video_sample):
    from ayase.modules.aigv_assessor import AIGVAssessorModule
    video_sample.quality_metrics = QualityMetrics()
    m = AIGVAssessorModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

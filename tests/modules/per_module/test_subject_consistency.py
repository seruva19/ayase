"""Tests for subject_consistency module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_subject_consistency_basics():
    from ayase.modules.subject_consistency import SubjectConsistencyModule
    _test_module_basics(SubjectConsistencyModule, "subject_consistency")

def test_subject_consistency_video(video_sample):
    from ayase.modules.subject_consistency import SubjectConsistencyModule
    video_sample.quality_metrics = QualityMetrics()
    m = SubjectConsistencyModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

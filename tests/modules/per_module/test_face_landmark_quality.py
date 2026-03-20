"""Tests for face_landmark_quality module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_face_landmark_quality_basics():
    from ayase.modules.face_landmark_quality import FaceLandmarkQualityModule
    _test_module_basics(FaceLandmarkQualityModule, "face_landmark_quality")

def test_face_landmark_quality_video(video_sample):
    from ayase.modules.face_landmark_quality import FaceLandmarkQualityModule
    video_sample.quality_metrics = QualityMetrics()
    m = FaceLandmarkQualityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

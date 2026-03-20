"""Tests for face_iqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_face_iqa_basics():
    from ayase.modules.face_iqa import FaceIQAModule
    _test_module_basics(FaceIQAModule, "face_iqa")

def test_face_iqa_image(image_sample):
    from ayase.modules.face_iqa import FaceIQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = FaceIQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_face_iqa_video(video_sample):
    from ayase.modules.face_iqa import FaceIQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = FaceIQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

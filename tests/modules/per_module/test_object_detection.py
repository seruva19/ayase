"""Tests for object_detection module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_object_detection_basics():
    from ayase.modules.object_detection import ObjectDetectionModule
    _test_module_basics(ObjectDetectionModule, "object_detection")

def test_object_detection_image(image_sample):
    from ayase.modules.object_detection import ObjectDetectionModule
    image_sample.quality_metrics = QualityMetrics()
    m = ObjectDetectionModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_object_detection_video(video_sample):
    from ayase.modules.object_detection import ObjectDetectionModule
    video_sample.quality_metrics = QualityMetrics()
    m = ObjectDetectionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

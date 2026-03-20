"""Tests for video_type_classifier module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_video_type_classifier_basics():
    from ayase.modules.video_type_classifier import VideoTypeClassifierModule
    _test_module_basics(VideoTypeClassifierModule, "video_type_classifier")

def test_video_type_classifier_image(image_sample):
    from ayase.modules.video_type_classifier import VideoTypeClassifierModule
    image_sample.quality_metrics = QualityMetrics()
    m = VideoTypeClassifierModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_video_type_classifier_video(video_sample):
    from ayase.modules.video_type_classifier import VideoTypeClassifierModule
    video_sample.quality_metrics = QualityMetrics()
    m = VideoTypeClassifierModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

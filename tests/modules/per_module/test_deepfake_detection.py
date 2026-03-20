"""Tests for deepfake_detection module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_deepfake_detection_basics():
    from ayase.modules.deepfake_detection import DeepfakeDetectionModule
    _test_module_basics(DeepfakeDetectionModule, "deepfake_detection")

def test_deepfake_detection_image(image_sample):
    from ayase.modules.deepfake_detection import DeepfakeDetectionModule
    image_sample.quality_metrics = QualityMetrics()
    m = DeepfakeDetectionModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_deepfake_detection_video(video_sample):
    from ayase.modules.deepfake_detection import DeepfakeDetectionModule
    video_sample.quality_metrics = QualityMetrics()
    m = DeepfakeDetectionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

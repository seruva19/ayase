"""Tests for scene_detection module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_scene_detection_basics():
    from ayase.modules.scene_detection import SceneDetectionModule
    _test_module_basics(SceneDetectionModule, "scene_detection")

def test_scene_detection_video(video_sample):
    from ayase.modules.scene_detection import SceneDetectionModule
    video_sample.quality_metrics = QualityMetrics()
    m = SceneDetectionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

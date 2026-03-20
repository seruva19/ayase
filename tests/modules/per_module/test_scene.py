"""Tests for scene module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_scene_basics():
    from ayase.modules.scene import SceneModule
    _test_module_basics(SceneModule, "scene")

def test_scene_video(video_sample):
    from ayase.modules.scene import SceneModule
    video_sample.quality_metrics = QualityMetrics()
    m = SceneModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

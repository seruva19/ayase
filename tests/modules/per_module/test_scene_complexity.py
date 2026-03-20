"""Tests for scene_complexity module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_scene_complexity_basics():
    from ayase.modules.scene_complexity import SceneComplexityModule
    _test_module_basics(SceneComplexityModule, "scene_complexity")

def test_scene_complexity_video(video_sample):
    from ayase.modules.scene_complexity import SceneComplexityModule
    video_sample.quality_metrics = QualityMetrics()
    m = SceneComplexityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

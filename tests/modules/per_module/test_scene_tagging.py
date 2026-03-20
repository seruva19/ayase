"""Tests for scene_tagging module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_scene_tagging_basics():
    from ayase.modules.scene_tagging import SceneTaggingModule
    _test_module_basics(SceneTaggingModule, "scene_tagging")

def test_scene_tagging_image(image_sample):
    from ayase.modules.scene_tagging import SceneTaggingModule
    image_sample.quality_metrics = QualityMetrics()
    m = SceneTaggingModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_scene_tagging_video(video_sample):
    from ayase.modules.scene_tagging import SceneTaggingModule
    video_sample.quality_metrics = QualityMetrics()
    m = SceneTaggingModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

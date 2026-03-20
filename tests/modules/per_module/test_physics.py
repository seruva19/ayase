"""Tests for physics module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_physics_basics():
    from ayase.modules.physics import PhysicsModule
    _test_module_basics(PhysicsModule, "physics")

def test_physics_video(video_sample):
    from ayase.modules.physics import PhysicsModule
    video_sample.quality_metrics = QualityMetrics()
    m = PhysicsModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

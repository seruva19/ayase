"""Tests for dynamics_controllability module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dynamics_controllability_basics():
    from ayase.modules.dynamics_controllability import DynamicsControllabilityModule
    _test_module_basics(DynamicsControllabilityModule, "dynamics_controllability")

def test_dynamics_controllability_video(video_sample):
    from ayase.modules.dynamics_controllability import DynamicsControllabilityModule
    video_sample.quality_metrics = QualityMetrics()
    m = DynamicsControllabilityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

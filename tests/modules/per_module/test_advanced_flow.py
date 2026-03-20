"""Tests for advanced_flow module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_advanced_flow_basics():
    from ayase.modules.advanced_flow import AdvancedFlowModule
    _test_module_basics(AdvancedFlowModule, "advanced_flow")

def test_advanced_flow_video(video_sample):
    from ayase.modules.advanced_flow import AdvancedFlowModule
    video_sample.quality_metrics = QualityMetrics()
    m = AdvancedFlowModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

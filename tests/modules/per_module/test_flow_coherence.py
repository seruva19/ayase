"""Tests for flow_coherence module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_flow_coherence_basics():
    from ayase.modules.flow_coherence import FlowCoherenceModule
    _test_module_basics(FlowCoherenceModule, "flow_coherence")

def test_flow_coherence_video(video_sample):
    from ayase.modules.flow_coherence import FlowCoherenceModule
    video_sample.quality_metrics = QualityMetrics()
    m = FlowCoherenceModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

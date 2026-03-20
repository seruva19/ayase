"""Tests for structural module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_structural_basics():
    from ayase.modules.structural import StructuralModule
    _test_module_basics(StructuralModule, "structural")

def test_structural_video(video_sample):
    from ayase.modules.structural import StructuralModule
    video_sample.quality_metrics = QualityMetrics()
    m = StructuralModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

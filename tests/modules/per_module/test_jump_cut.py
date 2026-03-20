"""Tests for jump_cut module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_jump_cut_basics():
    from ayase.modules.jump_cut import JumpCutModule
    _test_module_basics(JumpCutModule, "jump_cut")

def test_jump_cut_video(video_sample):
    from ayase.modules.jump_cut import JumpCutModule
    video_sample.quality_metrics = QualityMetrics()
    m = JumpCutModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

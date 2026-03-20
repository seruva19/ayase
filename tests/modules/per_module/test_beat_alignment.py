"""Tests for beat_alignment module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_beat_alignment_basics():
    from ayase.modules.beat_alignment import BeatAlignmentModule
    _test_module_basics(BeatAlignmentModule, "beat_alignment")

def test_beat_alignment_video(video_sample):
    from ayase.modules.beat_alignment import BeatAlignmentModule
    video_sample.quality_metrics = QualityMetrics()
    m = BeatAlignmentModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

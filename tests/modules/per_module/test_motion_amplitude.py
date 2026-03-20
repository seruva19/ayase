"""Tests for motion_amplitude module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_motion_amplitude_basics():
    from ayase.modules.motion_amplitude import MotionAmplitudeModule
    _test_module_basics(MotionAmplitudeModule, "motion_amplitude")

def test_motion_amplitude_video(video_sample):
    from ayase.modules.motion_amplitude import MotionAmplitudeModule
    video_sample.quality_metrics = QualityMetrics()
    m = MotionAmplitudeModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for flicker_detection module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_flicker_detection_basics():
    from ayase.modules.flicker_detection import FlickerDetectionModule
    _test_module_basics(FlickerDetectionModule, "flicker_detection")

def test_flicker_detection_video(video_sample):
    from ayase.modules.flicker_detection import FlickerDetectionModule
    video_sample.quality_metrics = QualityMetrics()
    m = FlickerDetectionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

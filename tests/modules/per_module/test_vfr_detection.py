"""Tests for vfr_detection module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vfr_detection_basics():
    from ayase.modules.vfr_detection import VFRDetectionModule
    _test_module_basics(VFRDetectionModule, "vfr_detection")

def test_vfr_detection_video(video_sample):
    from ayase.modules.vfr_detection import VFRDetectionModule
    video_sample.quality_metrics = QualityMetrics()
    m = VFRDetectionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for camera_jitter module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_camera_jitter_basics():
    from ayase.modules.camera_jitter import CameraJitterModule
    _test_module_basics(CameraJitterModule, "camera_jitter")

def test_camera_jitter_video(video_sample):
    from ayase.modules.camera_jitter import CameraJitterModule
    video_sample.quality_metrics = QualityMetrics()
    m = CameraJitterModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

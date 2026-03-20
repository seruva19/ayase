"""Tests for camera_motion module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_camera_motion_basics():
    from ayase.modules.camera_motion import CameraMotionModule
    _test_module_basics(CameraMotionModule, "camera_motion")

def test_camera_motion_video(video_sample):
    from ayase.modules.camera_motion import CameraMotionModule
    video_sample.quality_metrics = QualityMetrics()
    m = CameraMotionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

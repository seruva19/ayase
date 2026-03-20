"""Tests for action_recognition module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_action_recognition_basics():
    from ayase.modules.action_recognition import ActionRecognitionModule
    _test_module_basics(ActionRecognitionModule, "action_recognition")

def test_action_recognition_video(video_sample):
    from ayase.modules.action_recognition import ActionRecognitionModule
    video_sample.quality_metrics = QualityMetrics()
    m = ActionRecognitionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

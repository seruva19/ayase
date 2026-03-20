"""Tests for letterbox module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_letterbox_basics():
    from ayase.modules.letterbox import LetterboxModule
    _test_module_basics(LetterboxModule, "letterbox")

def test_letterbox_image(image_sample):
    from ayase.modules.letterbox import LetterboxModule
    image_sample.quality_metrics = QualityMetrics()
    m = LetterboxModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_letterbox_video(video_sample):
    from ayase.modules.letterbox import LetterboxModule
    video_sample.quality_metrics = QualityMetrics()
    m = LetterboxModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

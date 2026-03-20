"""Tests for celebrity_id module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_celebrity_id_basics():
    from ayase.modules.celebrity_id import CelebrityIDModule
    _test_module_basics(CelebrityIDModule, "celebrity_id")

def test_celebrity_id_image(image_sample):
    from ayase.modules.celebrity_id import CelebrityIDModule
    image_sample.quality_metrics = QualityMetrics()
    m = CelebrityIDModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_celebrity_id_video(video_sample):
    from ayase.modules.celebrity_id import CelebrityIDModule
    video_sample.quality_metrics = QualityMetrics()
    m = CelebrityIDModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

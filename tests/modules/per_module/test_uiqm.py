"""Tests for uiqm module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_uiqm_basics():
    from ayase.modules.uiqm import UIQMModule
    _test_module_basics(UIQMModule, "uiqm")

def test_uiqm_image(image_sample):
    from ayase.modules.uiqm import UIQMModule
    image_sample.quality_metrics = QualityMetrics()
    m = UIQMModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_uiqm_video(video_sample):
    from ayase.modules.uiqm import UIQMModule
    video_sample.quality_metrics = QualityMetrics()
    m = UIQMModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

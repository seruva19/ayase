"""Tests for cw_ssim module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_cw_ssim_basics():
    from ayase.modules.cw_ssim import CWSSIMModule
    _test_module_basics(CWSSIMModule, "cw_ssim")

def test_cw_ssim_image(image_sample):
    from ayase.modules.cw_ssim import CWSSIMModule
    image_sample.quality_metrics = QualityMetrics()
    m = CWSSIMModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_cw_ssim_video(video_sample):
    from ayase.modules.cw_ssim import CWSSIMModule
    video_sample.quality_metrics = QualityMetrics()
    m = CWSSIMModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

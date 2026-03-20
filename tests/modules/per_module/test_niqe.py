"""Tests for niqe module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_niqe_basics():
    from ayase.modules.niqe import NIQEModule
    _test_module_basics(NIQEModule, "niqe")

def test_niqe_image(image_sample):
    from ayase.modules.niqe import NIQEModule
    image_sample.quality_metrics = QualityMetrics()
    m = NIQEModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_niqe_video(video_sample):
    from ayase.modules.niqe import NIQEModule
    video_sample.quality_metrics = QualityMetrics()
    m = NIQEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

"""Tests for mdtvsfa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_mdtvsfa_basics():
    from ayase.modules.mdtvsfa import MDTVSFAModule
    _test_module_basics(MDTVSFAModule, "mdtvsfa")

def test_mdtvsfa_image(image_sample):
    from ayase.modules.mdtvsfa import MDTVSFAModule
    image_sample.quality_metrics = QualityMetrics()
    m = MDTVSFAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_mdtvsfa_video(video_sample):
    from ayase.modules.mdtvsfa import MDTVSFAModule
    video_sample.quality_metrics = QualityMetrics()
    m = MDTVSFAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

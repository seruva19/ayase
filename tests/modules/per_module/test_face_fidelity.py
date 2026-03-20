"""Tests for face_fidelity module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_face_fidelity_basics():
    from ayase.modules.face_fidelity import FaceFidelityModule
    _test_module_basics(FaceFidelityModule, "face_fidelity")

def test_face_fidelity_image(image_sample):
    from ayase.modules.face_fidelity import FaceFidelityModule
    image_sample.quality_metrics = QualityMetrics()
    m = FaceFidelityModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_face_fidelity_video(video_sample):
    from ayase.modules.face_fidelity import FaceFidelityModule
    video_sample.quality_metrics = QualityMetrics()
    m = FaceFidelityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

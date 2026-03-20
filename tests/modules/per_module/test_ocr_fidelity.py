"""Tests for ocr_fidelity module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ocr_fidelity_basics():
    from ayase.modules.ocr_fidelity import OCRFidelityModule
    _test_module_basics(OCRFidelityModule, "ocr_fidelity")

def test_ocr_fidelity_image(image_sample):
    from ayase.modules.ocr_fidelity import OCRFidelityModule
    image_sample.quality_metrics = QualityMetrics()
    m = OCRFidelityModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_ocr_fidelity_video(video_sample):
    from ayase.modules.ocr_fidelity import OCRFidelityModule
    video_sample.quality_metrics = QualityMetrics()
    m = OCRFidelityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

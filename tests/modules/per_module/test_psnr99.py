"""Tests for psnr99 module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics
from ayase.pipeline import Pipeline


def test_psnr99_basics():
    from ayase.modules.psnr99 import PSNR99Module
    _test_module_basics(PSNR99Module, "psnr99")


def test_psnr99_shared_image_video_field_is_adapted_and_opt_in():
    from ayase.modules.psnr99 import PSNR99Module

    assert PSNR99Module.field_provenance()["psnr99"] == "adapted"
    assert "psnr99" in Pipeline([PSNR99Module()])._provenance_excluded
    assert "psnr99" not in Pipeline(
        [PSNR99Module()], allow_provenance=["adapted"]
    )._provenance_excluded

def test_psnr99_no_reference(image_sample):
    from ayase.modules.psnr99 import PSNR99Module
    m = PSNR99Module()
    result = m.process(image_sample)
    assert result is image_sample

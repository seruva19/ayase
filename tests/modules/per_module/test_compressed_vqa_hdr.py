"""Tests for compressed_vqa_hdr module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_compressed_vqa_hdr_basics():
    from ayase.modules.compressed_vqa_hdr import CompressedVQAHDRModule
    _test_module_basics(CompressedVQAHDRModule, "compressed_vqa_hdr")

def test_compressed_vqa_hdr_no_reference(image_sample):
    from ayase.modules.compressed_vqa_hdr import CompressedVQAHDRModule
    m = CompressedVQAHDRModule()
    result = m.process(image_sample)
    assert result is image_sample

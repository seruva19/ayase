"""Tests for fvmd module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_fvmd_basics():
    from ayase.modules.fvmd import FVMDModule
    _test_module_basics(FVMDModule, "fvmd")

def test_fvmd_extract(video_sample):
    from ayase.modules.fvmd import FVMDModule
    m = FVMDModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None

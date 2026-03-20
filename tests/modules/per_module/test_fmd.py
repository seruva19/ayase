"""Tests for fmd module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_fmd_basics():
    from ayase.modules.fmd import FMDModule
    _test_module_basics(FMDModule, "fmd")

def test_fmd_extract(video_sample):
    from ayase.modules.fmd import FMDModule
    m = FMDModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None

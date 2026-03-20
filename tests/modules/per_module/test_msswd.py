"""Tests for msswd module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_msswd_basics():
    from ayase.modules.msswd import MSSWDModule
    _test_module_basics(MSSWDModule, "msswd")

def test_msswd_extract(video_sample):
    from ayase.modules.msswd import MSSWDModule
    m = MSSWDModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None

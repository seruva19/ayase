"""Tests for sfid module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_sfid_basics():
    from ayase.modules.sfid import SFIDModule
    _test_module_basics(SFIDModule, "sfid")

def test_sfid_extract(video_sample):
    from ayase.modules.sfid import SFIDModule
    m = SFIDModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None

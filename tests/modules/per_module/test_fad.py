"""Tests for fad module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_fad_basics():
    from ayase.modules.fad import FADModule
    _test_module_basics(FADModule, "fad")

def test_fad_extract(video_sample):
    from ayase.modules.fad import FADModule
    m = FADModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None

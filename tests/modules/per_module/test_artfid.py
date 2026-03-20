"""Tests for artfid module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_artfid_basics():
    from ayase.modules.artfid import ArtFIDModule
    _test_module_basics(ArtFIDModule, "artfid")

def test_artfid_no_reference(image_sample):
    from ayase.modules.artfid import ArtFIDModule
    m = ArtFIDModule()
    result = m.process(image_sample)
    assert result is image_sample

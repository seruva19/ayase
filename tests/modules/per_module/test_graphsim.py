"""Tests for graphsim module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_graphsim_basics():
    from ayase.modules.graphsim import GraphSIMModule
    _test_module_basics(GraphSIMModule, "graphsim")

def test_graphsim_no_reference(image_sample):
    from ayase.modules.graphsim import GraphSIMModule
    m = GraphSIMModule()
    result = m.process(image_sample)
    assert result is image_sample

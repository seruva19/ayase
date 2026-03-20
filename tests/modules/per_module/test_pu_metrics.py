"""Tests for pu_metrics module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_pu_metrics_basics():
    from ayase.modules.pu_metrics import PUMetricsModule
    _test_module_basics(PUMetricsModule, "pu_metrics")

def test_pu_metrics_no_reference(image_sample):
    from ayase.modules.pu_metrics import PUMetricsModule
    m = PUMetricsModule()
    result = m.process(image_sample)
    assert result is image_sample

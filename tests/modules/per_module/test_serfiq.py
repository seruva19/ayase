"""Tests for serfiq module."""

import importlib.util

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics

_HAS_MXNET = importlib.util.find_spec("mxnet") is not None
_HAS_INSIGHTFACE = importlib.util.find_spec("insightface") is not None


def test_serfiq_basics():
    from ayase.modules.serfiq import SERFIQModule
    _test_module_basics(SERFIQModule, "serfiq")

def test_serfiq_image(image_sample):
    from ayase.modules.serfiq import SERFIQModule
    image_sample.quality_metrics = QualityMetrics()
    m = SERFIQModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_serfiq_video(video_sample):
    from ayase.modules.serfiq import SERFIQModule
    video_sample.quality_metrics = QualityMetrics()
    m = SERFIQModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_serfiq_unavailable_without_mxnet(monkeypatch):
    """When MXNet is absent the module must degrade to _backend='unavailable'
    and must NOT fabricate a proxy score (no-heuristic policy)."""
    if _HAS_MXNET:
        pytest.skip("mxnet present; graceful-unavailable path not exercised here")
    from ayase.modules.serfiq import SERFIQModule
    from ayase.pipeline import PipelineModule

    # Exercise the real setup() body (the test suite runs with test_mode on,
    # which would otherwise early-return before backend resolution).
    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    prev = PipelineModule._global_test_mode
    PipelineModule.set_test_mode(False)
    try:
        m = SERFIQModule()
        m.setup()
    finally:
        PipelineModule.set_test_mode(prev)
    assert m._ml_available is False
    assert m._backend == "unavailable"


@pytest.mark.skipif(
    not (_HAS_MXNET and _HAS_INSIGHTFACE),
    reason="SER-FIQ real path requires both mxnet and insightface",
)
def test_serfiq_real_path(image_sample):
    """Real SER-FIQ path: loads the MXNet dropout-ArcFace + InsightFace, and if
    a face is detected produces a serfiq_score in [0, 1]."""
    from ayase.modules.serfiq import SERFIQModule
    m = SERFIQModule()
    m.setup()
    assert m._backend == "real"
    assert m._ml_available is True

    image_sample.quality_metrics = QualityMetrics()
    result = m.process(image_sample)
    assert result is image_sample
    score = result.quality_metrics.serfiq_score
    # None is acceptable if no face is found in the synthetic fixture; a real
    # score must fall in the SER-FIQ [0, 1] range.
    assert score is None or (0.0 <= score <= 1.0)

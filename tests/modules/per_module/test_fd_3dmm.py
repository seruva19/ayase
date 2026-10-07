"""Tests for fd_3dmm (FD + variation on 3DMM coefficients)."""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_fd_3dmm_basics():
    from ayase.modules.fd_3dmm import FD3DMMModule
    _test_module_basics(FD3DMMModule, "fd_3dmm")


def test_fd_3dmm_provenance():
    from ayase.modules.fd_3dmm import FD3DMMModule
    meta = FD3DMMModule.get_metadata()
    for f in ("fd_3dmm_expression", "fd_3dmm_pose", "expr_var_3dmm", "pose_var_3dmm"):
        assert meta["provenance"][f] == "adapted"
    for f in ("fd_3dmm_expression", "fd_3dmm_pose", "expr_var_3dmm", "pose_var_3dmm"):
        assert f in meta["deviations"]


def test_fd_3dmm_no_backend(tmp_path):
    from ayase.modules.fd_3dmm import FD3DMMModule
    m = FD3DMMModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.expr_var_3dmm is None


def test_fd_3dmm_variation_fields(tmp_path):
    from ayase.modules.fd_3dmm import FD3DMMModule
    m = FD3DMMModule()
    m._backend = "tddfa"
    rng = np.random.RandomState(1)
    params = rng.rand(30, 62).astype(np.float32)
    m._extractor = type("E", (), {
        "extract": staticmethod(lambda p: (params, np.arange(30)))})()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    out = m.process(sample)
    assert out.quality_metrics.expr_var_3dmm == pytest.approx(
        float(np.std(params[:, 52:62], axis=0).mean()))
    assert out.quality_metrics.pose_var_3dmm == pytest.approx(
        float(np.std(params[:, 0:12], axis=0).mean()))
    assert len(m._feature_cache) == 1


def test_fd_3dmm_no_reference_no_dataset_metric(tmp_path):
    from ayase.modules.fd_3dmm import FD3DMMModule
    m = FD3DMMModule()
    rng = np.random.RandomState(0)
    feats = [rng.rand(10, 62).astype(np.float32) for _ in range(3)]
    assert m.compute_distribution_metric(feats, None) is None


def test_frechet_identical_distributions_zero():
    from ayase.modules.fd_3dmm import _frechet_distance
    rng = np.random.RandomState(0)
    x = rng.randn(200, 5)
    mu, cov = x.mean(0), np.cov(x, rowvar=False)
    assert _frechet_distance(mu, cov, mu, cov) == pytest.approx(0.0, abs=1e-6)


def test_frechet_shifted_distributions_positive():
    from ayase.modules.fd_3dmm import _frechet_distance
    rng = np.random.RandomState(0)
    a = rng.randn(200, 5)
    b = a + 2.0
    fd = _frechet_distance(a.mean(0), np.cov(a, rowvar=False),
                           b.mean(0), np.cov(b, rowvar=False))
    assert fd > 0


def test_fd_3dmm_dataset_metrics_emitted(tmp_path):
    from ayase.modules.fd_3dmm import FD3DMMModule

    calls = {}

    class _Pipe:
        @staticmethod
        def add_dataset_metric(name, value):
            calls[name] = value

    m = FD3DMMModule()
    m.pipeline = _Pipe()
    rng = np.random.RandomState(0)
    gen = [rng.rand(40, 62).astype(np.float32) for _ in range(3)]
    ref = [rng.rand(40, 62).astype(np.float32) for _ in range(3)]
    m.compute_distribution_metric(gen, ref)
    assert set(calls) == {"fd_3dmm_expression", "fd_3dmm_pose"}
    assert all(v >= 0 for v in calls.values())

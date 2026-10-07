"""Focused correctness tests for the FID module."""

import sys

import numpy as np
import pytest

from ..conftest import _test_module_basics


def test_fid_basics():
    from ayase.modules.fid import FIDModule

    _test_module_basics(FIDModule, "fid")


def test_fid_is_adapted_for_mixed_image_video_input():
    from ayase.modules.fid import FIDModule

    assert FIDModule.field_provenance() == {"fid": "adapted"}


def test_fid_does_not_emit_non_frechet_fallback_without_scipy(monkeypatch):
    from ayase.modules.fid import FIDModule

    monkeypatch.setitem(sys.modules, "scipy", None)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([[2.0, 1.0], [1.0, 2.0]])

    assert FIDModule()._frechet_distance(x, y) is None


def test_fid_rejects_sqrtm_exception(monkeypatch):
    from scipy import linalg
    from ayase.modules.fid import FIDModule

    monkeypatch.setattr(linalg, "sqrtm", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad covariance")))
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert FIDModule()._frechet_distance(x, x) is None


def test_fid_retries_then_rejects_nonfinite_sqrtm(monkeypatch):
    from scipy import linalg
    from ayase.modules.fid import FIDModule

    calls = []

    def nonfinite(matrix, **kwargs):
        calls.append(matrix)
        return np.full(matrix.shape, np.nan), None

    monkeypatch.setattr(linalg, "sqrtm", nonfinite)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert FIDModule()._frechet_distance(x, x) is None
    assert len(calls) == 2


def test_fid_rejects_significant_complex_residue(monkeypatch):
    from scipy import linalg
    from ayase.modules.fid import FIDModule

    monkeypatch.setattr(
        linalg,
        "sqrtm",
        lambda matrix, **kwargs: (np.eye(matrix.shape[0], dtype=complex) * (1 + 0.01j), None),
    )
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert FIDModule()._frechet_distance(x, x) is None


def test_fid_valid_gaussian_distances():
    from ayase.modules.fid import FIDModule

    x = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    module = FIDModule()
    assert module._frechet_distance(x, x) == pytest.approx(0.0, abs=1e-9)
    assert module._frechet_distance(x, x + 2.0) == pytest.approx(8.0, abs=1e-8)

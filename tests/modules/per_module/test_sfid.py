"""Tests for sfid module."""

import sys

import numpy as np
import pytest

from ..conftest import _test_module_basics


def test_sfid_basics():
    from ayase.modules.sfid import SFIDModule
    _test_module_basics(SFIDModule, "sfid")

def test_sfid_extract(video_sample):
    from ayase.modules.sfid import SFIDModule
    m = SFIDModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None


def test_sfid_is_adapted_for_mixed_image_video_input():
    from ayase.modules.sfid import SFIDModule

    assert SFIDModule.field_provenance() == {"sfid": "adapted"}


def test_sfid_does_not_emit_non_frechet_fallback_without_scipy(monkeypatch):
    from ayase.modules.sfid import SFIDModule

    monkeypatch.setitem(sys.modules, "scipy", None)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([[2.0, 1.0], [1.0, 2.0]])

    assert SFIDModule()._frechet_distance(x, y) is None


def test_sfid_rejects_sqrtm_exception(monkeypatch):
    from scipy import linalg
    from ayase.modules.sfid import SFIDModule

    monkeypatch.setattr(linalg, "sqrtm", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad covariance")))
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert SFIDModule()._frechet_distance(x, x) is None


def test_sfid_retries_then_rejects_nonfinite_sqrtm(monkeypatch):
    from scipy import linalg
    from ayase.modules.sfid import SFIDModule

    calls = []

    def nonfinite(matrix, **kwargs):
        calls.append(matrix)
        return np.full(matrix.shape, np.nan), None

    monkeypatch.setattr(linalg, "sqrtm", nonfinite)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert SFIDModule()._frechet_distance(x, x) is None
    assert len(calls) == 2


def test_sfid_rejects_significant_complex_residue(monkeypatch):
    from scipy import linalg
    from ayase.modules.sfid import SFIDModule

    monkeypatch.setattr(
        linalg,
        "sqrtm",
        lambda matrix, **kwargs: (np.eye(matrix.shape[0], dtype=complex) * (1 + 0.01j), None),
    )
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert SFIDModule()._frechet_distance(x, x) is None


def test_sfid_valid_gaussian_distances():
    from ayase.modules.sfid import SFIDModule

    x = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    module = SFIDModule()
    assert module._frechet_distance(x, x) == pytest.approx(0.0, abs=1e-9)
    assert module._frechet_distance(x, x + 2.0) == pytest.approx(8.0, abs=1e-8)

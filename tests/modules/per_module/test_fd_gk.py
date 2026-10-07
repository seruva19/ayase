"""Tests for fd_gk (FD_g/FD_k on body pose + velocity distributions)."""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_fd_gk_basics():
    from ayase.modules.fd_gk import FDGestureModule
    _test_module_basics(FDGestureModule, "fd_gk")


def test_fd_gk_provenance():
    from ayase.modules.fd_gk import FDGestureModule
    meta = FDGestureModule.get_metadata()
    assert meta["provenance"]["fd_g"] == "adapted"
    assert meta["provenance"]["fd_k"] == "adapted"
    assert "Audio2Photoreal" in meta["sources"]["fd_g"]


def test_fd_gk_no_backend(tmp_path):
    from ayase.modules.fd_gk import FDGestureModule
    m = FDGestureModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample) is sample
    assert len(m._feature_cache) == 0


def test_fd_gk_no_reference_no_metric(tmp_path):
    from ayase.modules.fd_gk import FDGestureModule
    m = FDGestureModule()
    rng = np.random.RandomState(0)
    feats = [rng.rand(20, 132).astype(np.float32) for _ in range(3)]
    assert m.compute_distribution_metric(feats, None) is None


def test_fd_gk_dataset_metrics_emitted(tmp_path):
    from ayase.modules.fd_gk import FDGestureModule

    calls = {}

    class _Pipe:
        @staticmethod
        def add_dataset_metric(name, value):
            calls[name] = value

    m = FDGestureModule()
    m.pipeline = _Pipe()
    rng = np.random.RandomState(0)
    gen = [rng.rand(50, 132).astype(np.float32) for _ in range(3)]
    ref = [rng.rand(50, 132).astype(np.float32) for _ in range(3)]
    m.compute_distribution_metric(gen, ref)
    assert set(calls) == {"fd_g", "fd_k"}
    assert all(v >= 0 for v in calls.values())


def test_fd_gk_feature_shape(tmp_path, monkeypatch):
    from ayase.modules.fd_gk import FDGestureModule

    m = FDGestureModule()
    m._backend = "mediapipe"
    m._pose = object()
    seq = np.random.RandomState(0).rand(30, 33, 4).astype(np.float32)
    monkeypatch.setattr("ayase.modules._mp_seq.body_pose_seq",
                        lambda p, **kw: (seq, np.arange(30)))
    feats = m.extract_features(Sample(path=tmp_path / "v.mp4", is_video=True))
    assert feats.shape == (29, 132)

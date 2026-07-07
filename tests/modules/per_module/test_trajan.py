"""Tests for trajan module (pure-torch BootsTAPIR + track-autoencoder port)."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics
from ayase.pipeline import PipelineModule


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def _bootstapir_local() -> bool:
    """True if the BootsTAPIR checkpoint is already on disk (avoid test downloads)."""
    return Path("models/trajan/bootstapir_checkpoint_v2.pt").exists()


def _real_path_enabled() -> bool:
    """Only exercise the real (downloading) path when explicitly opted in."""
    return _has_torch() and (_bootstapir_local() or os.environ.get("AYASE_TRAJAN_REAL") == "1")


def test_trajan_basics():
    from ayase.modules.trajan import TRAJANModule
    _test_module_basics(TRAJANModule, "trajan")


def test_trajan_recover_tree():
    from ayase.modules.trajan import _recover_tree
    assert _recover_tree({"a/b": 1, "a/c": 2, "d": 3}) == {"a": {"b": 1, "c": 2}, "d": 3}


def test_trajan_threefry_deterministic():
    """The dither RNG must reproduce jax.random.uniform(PRNGKey(0)) bit-for-bit.

    These reference values come from jax 0.4.35 and are what makes the ported
    metric match the original TRAJAN score. If this drifts, the port is wrong.
    """
    from ayase.modules.trajan import _threefry_uniform
    u = _threefry_uniform((1, 128, 64))
    assert u.shape == (1, 128, 64)
    assert u.dtype == np.float32
    expected = np.array([0.86217594, 0.38886762, 0.5483135, 0.91096723, 0.45057428],
                        dtype=np.float32)
    np.testing.assert_allclose(u.reshape(-1)[:5], expected, atol=1e-6)
    assert 0.0 <= u.min() and u.max() < 1.0


@pytest.mark.skipif(not _has_torch(), reason="requires torch")
def test_trajan_backend_builds_and_runs():
    """Vendored torch defs build; the autoencoder runs with correct output shapes.

    Uses random weights (no checkpoint / no download) — this exercises the
    reimplemented Perceiver architecture and forward path cheaply.
    """
    import torch
    from ayase.modules.trajan import _build_backend
    be = _build_backend()
    assert hasattr(be, "TAPIR") and hasattr(be, "TrackAutoEncoder")
    ae = be.TrackAutoEncoder().eval()
    B, Qs, Qt, T = 1, 8, 4, 5
    inputs = {
        "support_tracks": torch.rand(B, Qs, T, 2),
        "support_tracks_visible": (torch.rand(B, Qs, T, 1) > 0.3).float(),
        "query_points": torch.cat([torch.randint(0, T, (B, Qt, 1)).float(),
                                   torch.rand(B, Qt, 2)], dim=-1),
        "boundary_frame": torch.tensor([T]),
    }
    with torch.no_grad():
        tracks, vis, cert = ae(inputs)
    assert tracks.shape == (B, Qt, 150, 2)
    assert vis.shape == (B, Qt, 150, 1)
    assert cert.shape == (B, Qt, 150, 1)


def test_trajan_video(video_sample):
    from ayase.modules.trajan import TRAJANModule
    video_sample.quality_metrics = QualityMetrics()
    m = TRAJANModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    # No-heuristic contract: real backend or None. In light test mode setup is
    # skipped (no downloads), so the field stays unset and no proxy is emitted.
    if m._backend != "port":
        assert m._ml_available is False
        assert m._backend in (None, "unavailable")
        assert result.quality_metrics.trajan_score is None


@pytest.mark.skipif(not _real_path_enabled(),
                    reason="requires torch + local BootsTAPIR checkpoint (or AYASE_TRAJAN_REAL=1)")
def test_trajan_real_backend(video_sample):
    """Guarded real-path test: pure-torch port validated to match the JAX original
    on reconstruction (max abs diff < 1e-3; see module docstring). Runs only where
    the checkpoints are already present, to avoid multi-hundred-MB test downloads."""
    from ayase.modules.trajan import TRAJANModule
    prev = PipelineModule._global_test_mode
    PipelineModule.set_test_mode(False)
    try:
        video_sample.quality_metrics = QualityMetrics()
        m = TRAJANModule({"max_frames": 8, "num_points": 512,
                          "num_support_tracks": 256, "num_target_tracks": 256,
                          "query_chunk_size": 64})
        m.on_mount()
        assert m._backend == "port"
        m.process(video_sample)
        score = video_sample.quality_metrics.trajan_score
        if score is not None:
            assert 0.0 <= score <= 1.0
    finally:
        PipelineModule.set_test_mode(prev)

"""Tests for stream_metric module."""

import importlib.util

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics
from ayase.pipeline import PipelineModule

_HAS_VSTREAM = importlib.util.find_spec("stream") is not None


@pytest.fixture
def no_test_mode(monkeypatch):
    """Disable the global light-mode flag so setup() exercises the real path."""
    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    prev = PipelineModule._global_test_mode
    PipelineModule.set_test_mode(False)
    yield
    PipelineModule.set_test_mode(prev)


def test_stream_metric_basics():
    from ayase.modules.stream_metric import STREAMModule
    _test_module_basics(STREAMModule, "stream_metric")


def test_stream_metric_extract(video_sample):
    from ayase.modules.stream_metric import STREAMModule
    m = STREAMModule()
    # No setup() called -> backend not initialised -> honest None, no crash.
    feat = m.extract_features(video_sample)
    assert feat is None
    assert video_sample is not None


def test_stream_metric_unavailable_when_backend_missing(monkeypatch, no_test_mode):
    """Graceful-unavailable contract: no v-stream/torch.hub -> unavailable, None."""
    from ayase.modules.stream_metric import STREAMModule

    import builtins

    m = STREAMModule()
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "stream" or name.startswith("stream."):
            raise ImportError("forced: no stream backend")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    m.setup()
    assert m._backend == "unavailable"
    assert m._stream_instance is None
    assert m.extract_features(SampleStub()) is None
    assert m.compute_distribution_metric([{"skewness": np.zeros(4)}], None) is None


class SampleStub:
    is_video = True
    path = "does-not-exist.mp4"


@pytest.mark.skipif(not _HAS_VSTREAM, reason="v-stream package not installed")
def test_stream_metric_real_path(video_sample, no_test_mode):
    """Real backend end-to-end. Skips if torch.hub cannot fetch the backbone."""
    from ayase.modules.stream_metric import STREAMModule

    m = STREAMModule({"num_frame": 16, "model": "swav"})
    m.setup()
    if m._backend != "v-stream":
        pytest.skip("torch.hub backbone unavailable (offline)")

    feat = m.extract_features(video_sample)
    assert feat is not None
    assert "skewness" in feat and "mean_signal" in feat
    assert feat["skewness"].shape[0] == m._stream_instance.num_embed
    assert feat["mean_signal"].shape[0] == m._stream_instance.num_embed

    # STREAM-T only needs a couple of videos; STREAM-S (prdc, k=5) needs more,
    # so it may return None here — that is acceptable. STREAM-T must be a float.
    feats = [feat, m.extract_features(video_sample)]
    refs = [m.extract_features(video_sample), m.extract_features(video_sample)]

    class _StatsCollector:
        def __init__(self):
            self.metrics = {}

        def add_dataset_metric(self, name, value):
            self.metrics[name] = value

    m.pipeline = _StatsCollector()
    m.compute_distribution_metric(feats, refs)
    assert "stream_temporal" in m.pipeline.metrics
    assert isinstance(m.pipeline.metrics["stream_temporal"], float)


@pytest.mark.skipif(not _HAS_VSTREAM, reason="v-stream package not installed")
def test_stream_metric_no_reference_returns_none():
    """Reference-based metric emits nothing without a real reference set."""
    from ayase.modules.stream_metric import STREAMModule

    m = STREAMModule()
    # Pretend the backend loaded so we exercise the no-reference branch.
    m._stream_instance = object()
    out = m.compute_distribution_metric([{"skewness": np.zeros(4), "mean_signal": np.zeros(4)}], None)
    assert out is None

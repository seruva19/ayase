"""Tests for dover module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dover_basics():
    from ayase.modules.dover import DOVERModule
    _test_module_basics(DOVERModule, "dover")

def test_dover_video(video_sample):
    from ayase.modules.dover import DOVERModule
    video_sample.quality_metrics = QualityMetrics()
    m = DOVERModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_native_dover_retries_amp_close_to_warning_threshold(monkeypatch):
    from contextlib import contextmanager

    import pytest

    torch = pytest.importorskip("torch")
    from ayase import runtime
    from ayase.modules import dover

    modes = []

    @contextmanager
    def inference_context(device, dtype, amp):
        modes.append((dtype, amp))
        yield

    def model(views, reduce_scores):
        value = 0.401 if modes[-1][1] else 0.397
        return [torch.tensor(value), torch.tensor(value)]

    def fuse(values):
        return dict(aesthetic=values[0], technical=values[1], overall=values[0])

    monkeypatch.setattr(runtime, "torch_inference_context", inference_context)
    monkeypatch.setattr(dover, "_fuse_results", fuse)
    module = dover.DOVERModule({"amp_enabled": True, "dtype": "bf16"})
    module._device = "cuda"
    module._model = model
    assert module._infer_native_views({}) == pytest.approx((0.397, 0.397, 0.397))
    assert modes == [("bf16", True), ("fp32", False)]


def test_native_dover_batch_pools_each_branch_per_video():
    import pytest

    torch = pytest.importorskip("torch")
    from ayase.modules.dover import DOVERModule

    module = DOVERModule()
    module._model = lambda views, **kwargs: list(views.values())
    videos = [
        {"aesthetic": torch.tensor([0.2]), "technical": torch.tensor([0.1, 0.2, 0.3])},
        {"aesthetic": torch.tensor([-0.2]), "technical": torch.tensor([-0.5, -0.1])},
    ]
    expected = [module._infer_native_views(video) for video in videos]
    assert module._infer_native_batch(videos) == expected

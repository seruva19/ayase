"""Regressions for audited FVD, KVD, and motion-smoothness edge cases."""

from pathlib import Path
import sys
import types
from typing import Optional

import numpy as np
import pytest

from ayase.models import Sample


def _video(name: str, reference: Optional[str] = None) -> Sample:
    return Sample(
        path=Path(name),
        is_video=True,
        reference_path=Path(reference) if reference else None,
    )


def test_fvd_subsample_budget_keeps_generated_reference_pairs_aligned(monkeypatch):
    from ayase.modules.fvd import FVDModule

    module = FVDModule({"subsample_videos": 2})
    monkeypatch.setattr(
        module,
        "_load_video_frames",
        lambda *_: np.zeros((16, 2, 2, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(module, "_extract_i3d_features", lambda frames: np.array([1.0, 2.0]))
    module._ml_available = True

    for index in range(3):
        module.process(_video(f"gen{index}.mp4", f"ref{index}.mp4"))

    assert len(module._feature_cache) == 2
    assert len(module._reference_cache) == 2
    assert module._processed_count == 2


def test_kvd_subsample_budget_keeps_generated_reference_pairs_aligned(monkeypatch):
    from ayase.modules.fvd import FVDModule
    from ayase.modules.kvd import KVDModule

    delegate = FVDModule({"subsample_videos": 1})
    monkeypatch.setattr(
        delegate,
        "_load_video_frames",
        lambda *_: np.zeros((16, 2, 2, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(delegate, "_extract_i3d_features", lambda frames: np.array([1.0, 2.0]))

    module = KVDModule()
    module._fvd_delegate = delegate
    module._ml_available = True
    module.process(_video("gen0.mp4", "ref0.mp4"))
    module.process(_video("gen1.mp4", "ref1.mp4"))

    assert len(module._feature_cache) == 1
    assert len(module._reference_cache) == 1


def test_distribution_metrics_reject_singleton_sets_without_nan():
    from ayase.modules.fvd import FVDModule
    from ayase.modules.kvd import KVDModule

    generated = [np.array([1.0, 2.0]), np.array([2.0, 3.0])]
    reference = [np.array([1.0, 2.0])]

    assert FVDModule().compute_distribution_metric(generated, reference) is None
    kvd = KVDModule()
    assert kvd.compute_distribution_metric(generated, reference) is None
    with pytest.raises(ValueError, match="at least two samples"):
        kvd._compute_mmd(np.stack(generated), np.stack(reference))


def test_fvd_supports_one_dimensional_feature_vectors():
    from ayase.modules.fvd import FVDModule

    score = FVDModule().compute_distribution_metric(
        [np.array([0.0]), np.array([1.0])],
        [np.array([0.0]), np.array([1.0])],
    )

    assert score == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("module_name", ["fvd", "kvd"])
def test_distribution_metrics_reject_nonfinite_and_malformed_features(module_name):
    if module_name == "fvd":
        from ayase.modules.fvd import FVDModule

        module = FVDModule()
    else:
        from ayase.modules.kvd import KVDModule

        module = KVDModule()

    valid = [np.array([0.0, 1.0]), np.array([1.0, 2.0])]
    nonfinite = [np.array([0.0, np.nan]), np.array([1.0, 2.0])]
    malformed = [np.array([0.0]), np.array([1.0, 2.0])]

    assert module.compute_distribution_metric(nonfinite, valid) is None
    assert module.compute_distribution_metric(malformed, valid) is None


@pytest.mark.parametrize("module_name", ["fvd", "kvd"])
def test_distribution_metrics_are_permutation_invariant(module_name):
    if module_name == "fvd":
        from ayase.modules.fvd import FVDModule

        module = FVDModule()
    else:
        from ayase.modules.kvd import KVDModule

        module = KVDModule()

    generated = [np.array([0.0, 1.0]), np.array([1.0, 3.0]), np.array([2.0, 0.0])]
    reference = [np.array([0.5, 2.0]), np.array([1.5, 1.0]), np.array([3.0, 4.0])]
    expected = module.compute_distribution_metric(generated, reference)
    permuted = module.compute_distribution_metric(generated[::-1], reference[1:] + reference[:1])

    assert expected is not None and np.isfinite(expected)
    assert permuted == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_fvd_linalg_failure_returns_none(monkeypatch):
    from scipy import linalg
    from ayase.modules.fvd import FVDModule

    monkeypatch.setattr(linalg, "sqrtm", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad covariance")))
    features = [np.array([0.0, 1.0]), np.array([1.0, 2.0])]

    assert FVDModule().compute_distribution_metric(features, features) is None


@pytest.mark.parametrize("module_name", ["fvd", "kvd"])
def test_invalid_distribution_is_not_published(module_name):
    if module_name == "fvd":
        from ayase.modules.fvd import FVDModule

        module = FVDModule()
    else:
        from ayase.modules.kvd import KVDModule

        module = KVDModule()

    published = []
    module.pipeline = types.SimpleNamespace(
        add_dataset_metric=lambda name, score: published.append((name, score))
    )
    module._feature_cache = [np.array([0.0, np.nan]), np.array([1.0, 2.0])]
    module._reference_cache = [np.array([0.0, 1.0]), np.array([1.0, 2.0])]

    module.on_dispose()

    assert published == []


def test_kvd_preserves_negative_unbiased_estimates():
    from ayase.modules.kvd import KVDModule

    features = np.array([[0.0], [1.0]])
    score = KVDModule()._compute_mmd(features, features)

    assert score == -3.5


def test_fvd_i3d_call_declares_preprocessed_input_contract():
    torch = __import__("torch")
    from ayase.modules.fvd import FVDModule

    calls = []

    class Detector:
        def __call__(self, inputs, **kwargs):
            calls.append((inputs, kwargs))
            return torch.tensor([[1.0, 2.0]])

    module = FVDModule()
    module.device = torch.device("cpu")
    module._r3d_model = Detector()
    result = module._extract_i3d_features(np.zeros((16, 224, 224, 3), dtype=np.uint8))

    assert result.tolist() == [1.0, 2.0]
    assert calls[0][1] == {"rescale": False, "resize": False, "return_features": True}
    assert tuple(calls[0][0].shape) == (1, 3, 16, 224, 224)


def test_i3d_urls_force_direct_download():
    from ayase.modules.fvd import FVDModule
    from ayase.modules.kvd import KVDModule

    assert FVDModule.models[0]["url"].endswith("?dl=1")
    assert KVDModule.models[0]["url"].endswith("?dl=1")


def test_fvd_setup_replaces_confirmed_cached_html(monkeypatch, tmp_path):
    torch = __import__("torch")
    from ayase import config
    from ayase.modules.fvd import FVDModule

    checkpoint = tmp_path / "fvd" / "i3d_torchscript.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"<!DOCTYPE html><html>Dropbox preview</html>")
    calls = []

    def fake_download(relative_path, url, models_dir):
        calls.append((relative_path, url, models_dir))
        if len(calls) == 2:
            checkpoint.write_bytes(b"PK\x03\x04torchscript archive")
        return checkpoint

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(config, "download_model_file", fake_download)
    monkeypatch.setattr(torch.jit, "load", lambda path, map_location: FakeModel())
    module = FVDModule({"models_dir": str(tmp_path)})
    module.device = torch.device("cpu")
    module._setup_i3d()

    assert len(calls) == 2
    assert all(call[1].endswith("?dl=1") for call in calls)
    assert checkpoint.read_bytes().startswith(b"PK")


def test_bundled_rife_moves_flownet_to_configured_device(monkeypatch, tmp_path):
    from ayase import runtime
    from ayase.modules.motion_smoothness import MotionSmoothnessModule

    moved_to = []

    class FlowNet:
        def to(self, device):
            moved_to.append(device)
            return self

    class FakeRIFE:
        def __init__(self):
            self.flownet = FlowNet()

        def load_model(self, path, rank):
            return None

        def eval(self):
            return None

    fake_module = types.ModuleType("ayase.third_party.rife.RIFE_HDv3")
    fake_module.Model = FakeRIFE
    fake_warplayer = types.ModuleType("ayase.third_party.rife.warplayer")
    fake_warplayer.device = __import__("torch").device("cuda")
    fake_warplayer.backwarp_tenGrid = {"stale": object()}
    monkeypatch.setitem(sys.modules, "ayase.third_party.rife.RIFE_HDv3", fake_module)
    monkeypatch.setitem(sys.modules, "ayase.third_party.rife.warplayer", fake_warplayer)
    monkeypatch.setattr(runtime, "resolve_torch_device", lambda configured: "cpu")
    monkeypatch.setattr("ayase.modules.motion_smoothness.os.path.exists", lambda path: True)

    module = MotionSmoothnessModule({"device": "cpu", "models_dir": str(tmp_path)})
    module.setup()

    assert moved_to == ["cpu"]
    assert fake_warplayer.device == __import__("torch").device("cpu")
    assert fake_warplayer.backwarp_tenGrid == {}
    assert module._rife_available


def test_custom_dispose_paths_run_pipeline_cleanup(monkeypatch):
    from ayase.modules.fvd import FVDModule
    from ayase.modules.kvd import KVDModule

    fvd = FVDModule()
    fvd_cleaned = []
    monkeypatch.setattr(fvd, "teardown", lambda: fvd_cleaned.append(True))
    fvd.on_dispose()

    delegate = FVDModule()
    delegate_cleaned = []
    monkeypatch.setattr(delegate, "teardown", lambda: delegate_cleaned.append(True))
    kvd = KVDModule()
    kvd._fvd_delegate = delegate
    kvd_cleaned = []
    monkeypatch.setattr(kvd, "teardown", lambda: kvd_cleaned.append(True))
    kvd.on_dispose()

    assert fvd_cleaned == [True]
    assert delegate_cleaned == [True]
    assert kvd_cleaned == [True]
    assert kvd._fvd_delegate is None

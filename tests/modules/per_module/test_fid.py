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
        return np.full(matrix.shape, np.nan)

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
        lambda matrix, **kwargs: np.eye(matrix.shape[0], dtype=complex) * (1 + 0.01j),
    )
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert FIDModule()._frechet_distance(x, x) is None


def test_fid_valid_gaussian_distances():
    from ayase.modules.fid import FIDModule

    x = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    module = FIDModule()
    assert module._frechet_distance(x, x) == pytest.approx(0.0, abs=1e-9)
    assert module._frechet_distance(x, x + 2.0) == pytest.approx(8.0, abs=1e-8)


@pytest.mark.parametrize("fail_batch", [False, True])
def test_official_fid_loader_and_batch_keep_generated_and_reference_sets(tmp_path, monkeypatch, fail_batch):
    from types import ModuleType

    import torch

    import ayase.modules.fid as fid
    from ayase.models import Sample

    calls = []

    class Inception(torch.nn.Module):
        BLOCK_INDEX_BY_DIM = {2048: 3}

        def __init__(self, *, output_blocks, resize_input):
            super().__init__()
            assert output_blocks == (3,)
            assert not resize_input

        def forward(self, inputs):
            assert torch.is_inference_mode_enabled()
            calls.append(len(inputs))
            if fail_batch and len(inputs) > 1:
                raise RuntimeError("batch failed")
            return [inputs]

    backend = ModuleType("pytorch_fid.inception")
    backend.InceptionV3 = Inception
    monkeypatch.setitem(sys.modules, backend.__name__, backend)

    def load(path, color):
        assert color == "rgb"
        return None if path.stem == "broken" else np.full((3, 3, 3), int(path.stem), np.uint8)

    monkeypatch.setattr(fid, "load_representative_frame", load)
    reference = tmp_path / "23.png"
    reference.touch()
    samples = [
        Sample(path=tmp_path / "broken.png", reference_path=reference, is_video=False),
        Sample(path=tmp_path / "5.png", reference_path=reference, is_video=False),
        Sample(path=tmp_path / "16.png", is_video=False),
    ]

    single = fid.FIDModule({"device": "cpu", "resize": 3})
    single.setup()
    for sample in samples:
        single.process(sample)
    calls.clear()
    batch = fid.FIDModule({"device": "cpu", "resize": 3})
    batch.setup()
    assert batch.process_batch(samples) == samples
    assert len(batch._feature_cache) == len(batch._reference_cache) == 2
    np.testing.assert_array_equal(batch._feature_cache, single._feature_cache)
    np.testing.assert_array_equal(batch._reference_cache, single._reference_cache)
    assert calls == ([2, 1, 1, 1, 1] if fail_batch else [2, 2])

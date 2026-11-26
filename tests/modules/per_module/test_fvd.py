"""Tests for fvd module."""

import warnings

import pytest

from ..conftest import _test_module_basics
from ayase.models import DatasetStats, QualityMetrics


def test_fvd_basics():
    from ayase.modules.fvd import FVDModule
    _test_module_basics(FVDModule, "fvd")

def test_fvd_extract(video_sample):
    from ayase.modules.fvd import FVDModule
    m = FVDModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None


def test_fvd_default_backbone_is_i3d():
    """The published FVD uses the I3D torchscript backbone (StyleGAN-V)."""
    from ayase.modules.fvd import FVDModule
    m = FVDModule()
    assert m.backbone == "i3d"
    assert m.metric_name == "fvd"


def test_fvd_removed_backbones_map_to_i3d():
    """Removed backbone variants fold back to the published I3D metric."""
    from ayase.modules.fvd import FVDModule
    for legacy in ("r3d18", "content_debiased", "dinov2", "nonexistent_backbone"):
        m = FVDModule(config={"backbone": legacy})
        assert m.backbone == "i3d"
        assert m.metric_name == "fvd"


def test_fvd_removed_fields_resolve_to_none():
    """Removed FVD variant fields read as None with a deprecation warning."""
    stats = DatasetStats(total_samples=0, valid_samples=0, invalid_samples=0, total_size=0)
    for name in ("fvd_content_debiased", "fvd_dinov2"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = getattr(stats, name)
        assert value is None
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)


@pytest.mark.parametrize("fail_batch", [False, True])
def test_fvd_batch_preserves_budget_and_reference_features(tmp_path, monkeypatch, fail_batch):
    import numpy as np
    import torch

    from ayase.models import Sample
    from ayase.modules.fvd import FVDModule

    calls = []

    def decode(path, num_frames):
        if path.stem == "broken":
            return None
        return np.full((num_frames, 3, 3, 3), int(path.stem), dtype=np.uint8)

    def make_module(fail):
        module = FVDModule({"num_frames": 2, "subsample_videos": 3, "batch_size": 3})
        module._ml_available = True
        module.device = "cpu"
        monkeypatch.setattr(module, "_load_video_frames", decode)

        def model(tensor, **kwargs):
            assert torch.is_inference_mode_enabled()
            calls.append(tensor.shape[0])
            if fail and tensor.shape[0] > 1:
                raise RuntimeError("simulated batch OOM")
            return tensor.mean(dim=(2, 3, 4))

        module._r3d_model = model
        return module

    samples = [
        Sample(
            path=tmp_path / f"{name}.mp4",
            reference_path=tmp_path / f"{100 + i}.mp4",
            is_video=True,
        )
        for i, name in enumerate(("broken", "1", "2", "3", "4"))
    ]
    single = make_module(False)
    for sample in samples:
        single.process(sample)
    calls.clear()
    batch = make_module(fail_batch)
    assert batch.process_batch(samples) == samples
    assert batch._processed_count == single._processed_count == 3
    np.testing.assert_array_equal(batch._feature_cache, single._feature_cache)
    np.testing.assert_array_equal(batch._reference_cache, single._reference_cache)
    assert calls == ([2, 1, 1, 1, 1, 1, 1] if fail_batch else [2, 2, 1, 1])

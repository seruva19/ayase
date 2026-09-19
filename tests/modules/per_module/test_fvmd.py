"""Focused offline tests for the official-compatible FVMD runtime."""

import numpy as np
import pytest

from tests.modules.conftest import _test_module_basics


def test_fvmd_basics_and_applicability_header():
    import ayase.modules.fvmd as module

    _test_module_basics(module.FVMDModule, "fvmd")
    doc = module.__doc__.lower()
    assert "dataset-level" in doc
    assert "paired-video fidelity score" in doc
    assert "motion manner" in doc
    assert "official_fvmd_1_0_0_compat" in doc


def test_released_velocity_and_acceleration_conventions():
    import torch

    from ayase.vendor.fvmd_official import calc_acceleration, calc_velocity

    trajectory = torch.tensor([[[[0.0, 0.0]], [[1.0, 2.0]], [[3.0, 6.0]], [[6.0, 12.0]]]])
    velocity = calc_velocity(trajectory)
    acceleration_channel = calc_acceleration(trajectory)

    assert velocity[:, :, 0].tolist() == [[[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]]
    assert acceleration_channel[:, :, 0].tolist() == [
        [[0.0, 0.0], [0.0, 0.0], [2.0, 4.0], [3.0, 6.0]]
    ]


def test_histogram_feature_shape_and_static_zero():
    from ayase.vendor.fvmd_official import combine_motion_histograms

    field = np.zeros((2, 16, 400, 2), dtype=np.float32)
    features = combine_motion_histograms(field, field)

    assert features.shape == (2, 1024)
    assert np.count_nonzero(features) == 0


def test_distribution_distance_identity_and_shift():
    from ayase.modules.fvmd import FVMDModule

    rng = np.random.default_rng(20260920)
    reference = rng.normal(size=(8, 6))
    module = FVMDModule()

    identity = module.compute_distribution_metric([reference.copy()], [reference])
    changed = module.compute_distribution_metric([reference + 2.0], [reference])

    assert identity == pytest.approx(0.0, abs=1e-8)
    assert changed > identity


def test_distribution_requires_real_reference_and_two_windows():
    from ayase.modules.fvmd import FVMDModule

    module = FVMDModule()
    with pytest.raises(ValueError, match="separate reference"):
        module.compute_distribution_metric([np.zeros((2, 4))])
    with pytest.raises(ValueError, match="at least two"):
        module.compute_distribution_metric([np.zeros((1, 4))], [np.zeros((1, 4))])


def test_official_sliding_window_starts_and_uniform_limit():
    from ayase.modules.fvmd import FVMDModule

    assert FVMDModule()._window_starts(18).tolist() == [0, 1, 2]
    limited = FVMDModule({"max_windows_per_video": 2})
    assert limited._window_starts(20).tolist() == [0, 4]


def test_window_feature_extraction_uses_official_1024d_layout(monkeypatch):
    import torch

    import ayase.vendor.fvmd_official as official
    from ayase.modules.fvmd import FVMDModule

    def fake_tracking(_model, tensor, N, iters):
        assert tensor.shape == (1, 16, 3, 256, 256)
        assert N == 400
        assert iters == 16
        time = torch.arange(16, dtype=torch.float32).view(1, 16, 1, 1)
        result = torch.zeros((1, 16, 400, 2), dtype=torch.float32)
        result[..., 0:1] = time
        return result

    monkeypatch.setattr(official, "run_tracking", fake_tracking)
    module = FVMDModule({"max_windows_per_video": 2})
    module.device = torch.device("cpu")
    module._model = object()
    frames = np.zeros((18, 256, 256, 3), dtype=np.uint8)

    result = module._extract_window_features(frames)

    assert result.shape == (2, 1024)
    assert np.isfinite(result).all()


def test_extract_without_setup_is_gracefully_unavailable(video_sample):
    from ayase.modules.fvmd import FVMDModule

    module = FVMDModule()
    assert module.extract_features(video_sample) is None
    assert module._backend == "unavailable"

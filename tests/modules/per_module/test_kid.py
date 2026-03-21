"""Tests for KID (Kernel Inception Distance) module."""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, DatasetStats


def test_kid_basics():
    from ayase.modules.kid import KIDModule

    _test_module_basics(KIDModule, "kid")


def test_kid_skip_without_ml(image_sample):
    from ayase.modules.kid import KIDModule

    m = KIDModule()
    # Don't call setup — _ml_available stays False
    result = m.process(image_sample)
    assert result is image_sample


def test_kid_feature_extraction(image_sample):
    from ayase.modules.kid import KIDModule

    m = KIDModule()
    m.setup()
    if not m._ml_available:
        pytest.skip("ML dependencies not available")
    feat = m.extract_features(image_sample)
    # For native backend, should return a numpy array
    if m._backend == "native":
        assert isinstance(feat, np.ndarray)
        assert feat.shape == (2048,)
    else:
        # For cleanfid/torch-fidelity, returns path string
        assert feat is not None


def test_kid_compute_same_distribution():
    """KID between two identical feature sets should be approximately 0."""
    from ayase.modules.kid import KIDModule

    m = KIDModule({"subset_size": 10, "num_subsets": 10})
    # Synthetic features
    np.random.seed(42)
    features = np.random.randn(50, 2048).astype(np.float32)

    kid_mean, kid_std = m._compute_kid_mmd(features, features)
    # KID of same distribution should be close to 0
    assert abs(kid_mean) < 0.5, f"KID of same distribution too large: {kid_mean}"


def test_kid_compute_different_distributions():
    """KID between different distributions should be > 0."""
    from ayase.modules.kid import KIDModule

    m = KIDModule({"subset_size": 10, "num_subsets": 10})
    np.random.seed(42)
    features_a = np.random.randn(50, 2048).astype(np.float32)
    features_b = np.random.randn(50, 2048).astype(np.float32) + 5.0  # Shifted

    kid_mean, _ = m._compute_kid_mmd(features_a, features_b)
    assert kid_mean > 0, f"KID should be > 0 for different distributions, got {kid_mean}"


def test_kid_dataset_stats_field():
    """Verify kid and kid_std fields exist in DatasetStats."""
    stats = DatasetStats(
        total_samples=10,
        valid_samples=10,
        invalid_samples=0,
        total_size=1000,
    )
    assert hasattr(stats, "kid")
    assert hasattr(stats, "kid_std")
    stats.kid = 0.05
    stats.kid_std = 0.01
    assert stats.kid == 0.05
    assert stats.kid_std == 0.01

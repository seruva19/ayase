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
        pytest.skip("clean-fid / torch-fidelity not installed")
    feat = m.extract_features(image_sample)
    # Published backends work on file paths; features come back at batch time.
    assert isinstance(feat, str)


def test_kid_unavailable_without_backends():
    """Without a published backend the module stays unavailable (no substitute)."""
    import sys
    from unittest import mock
    from ayase.modules.kid import KIDModule

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in ("cleanfid", "torch_fidelity"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=blocked):
        m = KIDModule()
        m.setup()
    assert m._backend == "unavailable"
    assert not m._ml_available


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

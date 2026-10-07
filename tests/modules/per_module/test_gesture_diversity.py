"""Tests for gesture_diversity (EMAGE L1 Diversity on MediaPipe pose)."""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_gesture_diversity_basics():
    from ayase.modules.gesture_diversity import GestureDiversityModule
    _test_module_basics(GestureDiversityModule, "gesture_diversity")


def test_gesture_diversity_provenance():
    from ayase.modules.gesture_diversity import GestureDiversityModule
    meta = GestureDiversityModule.get_metadata()
    assert meta["provenance"]["l1_diversity"] == "adapted"
    assert "EMAGE" in meta["sources"]["l1_diversity"]


def test_gesture_diversity_too_few_samples():
    from ayase.modules.gesture_diversity import GestureDiversityModule
    m = GestureDiversityModule()
    feats = [np.ones(66, dtype=np.float32)]
    assert m.compute_distribution_metric(feats, None) is None


def test_gesture_diversity_identical_zero():
    from ayase.modules.gesture_diversity import GestureDiversityModule
    m = GestureDiversityModule()
    feats = [np.ones(66, dtype=np.float32)] * 4
    assert m.compute_distribution_metric(feats, None) == pytest.approx(0.0)


def test_gesture_diversity_distinct_positive():
    from ayase.modules.gesture_diversity import GestureDiversityModule

    calls = {}

    class _Pipe:
        @staticmethod
        def add_dataset_metric(name, value):
            calls[name] = value

    m = GestureDiversityModule()
    m.pipeline = _Pipe()
    rng = np.random.RandomState(0)
    feats = [rng.rand(66).astype(np.float32) * (i + 1) for i in range(4)]
    score = m.compute_distribution_metric(feats, None)
    assert score > 0
    assert calls["l1_diversity"] == pytest.approx(score)


def test_gesture_diversity_no_backend(tmp_path):
    from ayase.modules.gesture_diversity import GestureDiversityModule
    m = GestureDiversityModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    assert m.extract_features(sample) is None

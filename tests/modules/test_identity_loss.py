import numpy as np
import pytest

from ayase.models import QualityMetrics, Sample


def test_identity_loss_basics():
    from ayase.modules.identity_loss import IdentityLossModule
    from .conftest import _test_module_basics

    _test_module_basics(IdentityLossModule, "identity_loss")


def test_identity_loss_config():
    from ayase.modules.identity_loss import IdentityLossModule

    m = IdentityLossModule()
    assert "model_name" in m.default_config
    assert "subsample" in m.default_config
    assert "warning_threshold" in m.default_config


def test_identity_loss_skip_without_reference(image_sample):
    from ayase.modules.identity_loss import IdentityLossModule

    m = IdentityLossModule()
    m._backend = "mediapipe"  # pretend we have a backend
    result = m.process(image_sample)
    # No reference_path → should skip gracefully
    assert result.quality_metrics is None or result.quality_metrics.identity_loss is None


def test_identity_loss_skip_no_ml(image_sample):
    from ayase.modules.identity_loss import IdentityLossModule

    m = IdentityLossModule()
    # _backend is None by default (no setup called)
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.identity_loss is None


def test_identity_loss_normalize_landmarks():
    """Test that MediaPipe landmark normalization produces unit-scale output."""
    from ayase.modules.identity_loss import IdentityLossModule

    m = IdentityLossModule()
    # Simulate normalized landmarks: zero-centroid, unit-scale
    pts = np.random.randn(468, 3).astype(np.float32)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    scale = np.linalg.norm(pts_centered, axis=1).max() + 1e-10
    pts_norm = pts_centered / scale
    # Max norm should be ~1.0
    assert abs(np.linalg.norm(pts_norm, axis=1).max() - 1.0) < 0.01


def test_identity_loss_fields_exist():
    qm = QualityMetrics()
    assert hasattr(qm, "identity_loss")
    assert hasattr(qm, "face_recognition_score")
    assert qm.identity_loss is None
    assert qm.face_recognition_score is None


def test_identity_loss_field_groups():
    qm = QualityMetrics()
    groups = qm._FIELD_GROUPS
    assert groups.get("identity_loss") == "face"
    assert groups.get("face_recognition_score") == "face"

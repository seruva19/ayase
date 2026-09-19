"""Deterministic offline tests for PoseHeat-SSIM."""

import numpy as np
import sys
import types

from ayase.models import Sample
from tests.modules.conftest import _test_module_basics


def _pose() -> np.ndarray:
    points = np.zeros((133, 2), dtype=np.float64)
    points[:6] = np.array(
        [[12, 10], [22, 12], [17, 22], [10, 31], [25, 32], [18, 43]],
        dtype=np.float64,
    )
    return points


def _scores(value: float = 1.0) -> np.ndarray:
    scores = np.zeros(133, dtype=np.float64)
    scores[:6] = value
    return scores


def test_pose_heat_ssim_basics():
    from ayase.modules.pose_heat_ssim import PoseHeatSSIMModule

    _test_module_basics(PoseHeatSSIMModule, "pose_heat_ssim")


def test_setup_normalizes_auto_device_to_rtmlib_name(monkeypatch):
    from ayase.modules.pose_heat_ssim import PoseHeatSSIMModule
    import ayase.runtime

    monkeypatch.setitem(sys.modules, "rtmlib", types.SimpleNamespace(Wholebody=object))
    monkeypatch.setattr(ayase.runtime, "resolve_torch_device", lambda _name: "cuda:7")
    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    monkeypatch.setattr(PoseHeatSSIMModule, "_global_test_mode", False)
    module = PoseHeatSSIMModule()
    module.setup()
    assert module._backend_available is True
    assert module._device == "cuda"


def test_identical_pose_scores_one():
    from ayase.modules.pose_heat_ssim import pose_heat_ssim_score

    pose = _pose()
    score = pose_heat_ssim_score(pose, _scores(), pose, _scores(), (64, 64))
    assert score == 1.0


def test_translation_and_scale_are_removed():
    from ayase.modules.pose_heat_ssim import pose_heat_ssim_score

    reference = _pose()
    generated = 1.35 * reference + np.array([6.0, -3.0])
    score = pose_heat_ssim_score(reference, _scores(), generated, _scores(), (96, 96))
    assert score is not None
    assert score == 1.0


def test_nonrigid_joint_perturbation_reduces_score():
    from ayase.modules.pose_heat_ssim import pose_heat_ssim_score

    reference = _pose()
    generated = reference.copy()
    generated[4] += np.array([13.0, -8.0])
    score = pose_heat_ssim_score(reference, _scores(), generated, _scores(), (64, 64))
    assert score is not None
    assert 0.0 <= score < 1.0


def test_only_jointly_confident_joints_are_rendered():
    from ayase.modules.pose_heat_ssim import pose_heat_ssim_score

    reference = _pose()
    generated = reference.copy()
    generated[5] = [60.0, 60.0]
    generated_scores = _scores()
    generated_scores[5] = 0.1
    score = pose_heat_ssim_score(
        reference,
        _scores(),
        generated,
        generated_scores,
        (64, 64),
        confidence_threshold=0.3,
    )
    assert score == 1.0


def test_insufficient_joint_overlap_is_unset():
    from ayase.modules.pose_heat_ssim import pose_heat_ssim_score

    generated_scores = np.zeros(133)
    generated_scores[:2] = 1.0
    assert (
        pose_heat_ssim_score(_pose(), _scores(), _pose(), generated_scores, (64, 64))
        is None
    )


def test_multi_person_output_is_rejected():
    from ayase.modules.pose_heat_ssim import single_wholebody_pose

    keypoints = np.stack([_pose(), _pose() + 2.0])
    scores = np.stack([_scores(), _scores()])
    assert single_wholebody_pose((keypoints, scores)) is None
    single = single_wholebody_pose((keypoints[:1], scores[:1]))
    assert single is not None
    assert single[0].shape == (133, 2)


def test_process_gracefully_returns_same_sample_without_reference(tmp_path):
    from ayase.modules.pose_heat_ssim import PoseHeatSSIMModule

    sample = Sample(path=tmp_path / "missing.mp4", is_video=True)
    module = PoseHeatSSIMModule()
    module._wholebody = lambda frame: (_pose()[None, ...], _scores()[None, ...])
    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_process_gracefully_returns_same_sample_when_backend_unavailable(tmp_path):
    from ayase.modules.pose_heat_ssim import PoseHeatSSIMModule

    sample = Sample(
        path=tmp_path / "missing.mp4",
        reference_path=tmp_path / "reference.mp4",
        is_video=True,
    )
    module = PoseHeatSSIMModule()
    assert module.process(sample) is sample
    assert sample.quality_metrics is None

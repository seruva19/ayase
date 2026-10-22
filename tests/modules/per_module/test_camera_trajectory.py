"""Tests for the camera_trajectory module (CamI2V RotErr/TransErr/CamMC)."""

import json

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def _make_sample(tmp_path, name="clip.mp4"):
    """A video Sample in an isolated dir (no decodable content needed for
    trajectory loading — only the sidecar next to the path is read)."""
    video = tmp_path / name
    video.write_bytes(b"")
    return Sample(path=video, is_video=True)


def _rot_z(theta: float) -> np.ndarray:
    """4x4 homogeneous rotation about +z by *theta* radians (zero translation)."""
    c, s = np.cos(theta), np.sin(theta)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return m


def _translate(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = (x, y, z)
    return m


# ---------------------------------------------------------------------------
# Module basics
# ---------------------------------------------------------------------------


def test_camera_trajectory_basics():
    from ayase.modules.camera_trajectory import CameraTrajectoryModule

    _test_module_basics(CameraTrajectoryModule, "camera_trajectory")


@pytest.mark.parametrize(
    ("feature_help", "matching_help", "expected"),
    [
        (
            "--SiftExtraction.use_gpu arg (=1)",
            "--SiftMatching.use_gpu arg (=1)",
            ("--SiftExtraction.use_gpu", "--SiftMatching.use_gpu"),
        ),
        (
            "--FeatureExtraction.use_gpu arg (=1)",
            "--FeatureMatching.use_gpu arg (=1)",
            ("--FeatureExtraction.use_gpu", "--FeatureMatching.use_gpu"),
        ),
    ],
)
def test_colmap_gpu_options_follow_advertised_cli(
    monkeypatch, feature_help, matching_help, expected
):
    import subprocess
    import ayase.modules.camera_trajectory as ct

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        output = feature_help if cmd[1] == "feature_extractor" else matching_help
        return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")

    ct._COLMAP_GPU_OPTIONS.clear()
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert ct._colmap_gpu_options("configured-colmap.exe") == expected
    assert ct._colmap_gpu_options("configured-colmap.exe") == expected
    assert calls == [
        ["configured-colmap.exe", "feature_extractor", "-h"],
        ["configured-colmap.exe", "sequential_matcher", "-h"],
    ]


# ---------------------------------------------------------------------------
# Pure math: RotErr / TransErr / CamMC on known simple cases
# ---------------------------------------------------------------------------


def test_rotation_geodesic_matches_angle():
    from ayase.modules.camera_trajectory import _rotation_geodesic

    theta = np.pi / 3
    ang = _rotation_geodesic(np.eye(3), _rot_z(theta)[:3, :3])
    assert ang == pytest.approx(theta, abs=1e-9)


def test_identical_trajectories_have_zero_error():
    from ayase.modules.camera_trajectory import compute_trajectory_errors

    traj = np.stack([np.eye(4), _translate(1.0), _translate(2.0, 0.5)])
    errs = compute_trajectory_errors(traj, traj)
    assert errs["rot_err"] == pytest.approx(0.0, abs=1e-6)
    assert errs["trans_err"] == pytest.approx(0.0, abs=1e-6)
    assert errs["cammc"] == pytest.approx(0.0, abs=1e-6)


def test_translation_error_is_scale_invariant():
    # A scaled-up estimated trajectory should incur ~0 TransErr after the
    # normalize_t scale alignment (translations divided by their own max norm).
    from ayase.modules.camera_trajectory import compute_trajectory_errors

    target = np.stack([np.eye(4), _translate(1.0), _translate(2.0)])
    estimated = np.stack([np.eye(4), _translate(2.0), _translate(4.0)])
    errs = compute_trajectory_errors(estimated, target)
    assert errs["rot_err"] == pytest.approx(0.0, abs=1e-6)
    assert errs["trans_err"] == pytest.approx(0.0, abs=1e-6)


def test_known_rotation_error_in_radians():
    from ayase.modules.camera_trajectory import compute_trajectory_errors

    target = np.stack([np.eye(4), np.eye(4), np.eye(4)])
    # 30-degree rotation on the middle relative pose only.
    estimated = np.stack([np.eye(4), _rot_z(np.radians(30.0)), np.eye(4)])
    errs = compute_trajectory_errors(estimated, target)
    assert errs["rot_err"] == pytest.approx(np.radians(30.0), abs=1e-6)
    assert errs["trans_err"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Target-trajectory convention parsing
# ---------------------------------------------------------------------------


def test_trajectory_sidecar_list_form(tmp_path):
    from ayase.modules.camera_trajectory import CameraTrajectoryModule

    sample = _make_sample(tmp_path)
    traj = [np.eye(4).tolist(), _translate(1.0).tolist(), _translate(2.0).tolist()]
    sidecar = tmp_path / (sample.path.stem + ".camera.json")
    sidecar.write_text(json.dumps(traj), encoding="utf-8")

    m = CameraTrajectoryModule()
    loaded = m._load_target_trajectory(sample)
    assert loaded is not None
    assert loaded.shape == (3, 4, 4)
    np.testing.assert_allclose(loaded[1], _translate(1.0))


def test_trajectory_sidecar_dict_form_with_key(tmp_path):
    from ayase.modules.camera_trajectory import CameraTrajectoryModule

    sample = _make_sample(tmp_path)
    traj = [np.eye(4).tolist(), _translate(1.0).tolist()]
    sidecar = tmp_path / (sample.path.stem + ".camera.json")
    sidecar.write_text(json.dumps({"camera_trajectory": traj}), encoding="utf-8")

    m = CameraTrajectoryModule()
    loaded = m._load_target_trajectory(sample)
    assert loaded is not None and loaded.shape == (2, 4, 4)


def test_trajectory_accepts_3x4_matrices(tmp_path):
    from ayase.modules.camera_trajectory import CameraTrajectoryModule

    sample = _make_sample(tmp_path)
    traj = [np.eye(4)[:3].tolist(), _translate(1.0)[:3].tolist()]
    sidecar = tmp_path / (sample.path.stem + ".camera.json")
    sidecar.write_text(json.dumps(traj), encoding="utf-8")

    m = CameraTrajectoryModule()
    loaded = m._load_target_trajectory(sample)
    assert loaded is not None and loaded.shape == (2, 4, 4)
    np.testing.assert_allclose(loaded[0], np.eye(4))


def test_missing_trajectory_returns_none(tmp_path):
    from ayase.modules.camera_trajectory import CameraTrajectoryModule

    sample = _make_sample(tmp_path)
    m = CameraTrajectoryModule()
    assert m._load_target_trajectory(sample) is None


# ---------------------------------------------------------------------------
# End-to-end process() with a monkeypatched (real-backend-free) pose estimator
# ---------------------------------------------------------------------------


def test_process_with_mocked_pose_estimator(tmp_path):
    from ayase.modules.camera_trajectory import CameraTrajectoryModule

    sample = _make_sample(tmp_path)
    traj = np.stack([np.eye(4), _translate(1.0), _translate(2.0, 0.5)])
    sidecar = tmp_path / (sample.path.stem + ".camera.json")
    sidecar.write_text(json.dumps(traj.tolist()), encoding="utf-8")

    m = CameraTrajectoryModule()
    m._ml_available = True
    m._backend = "vggt"
    # Pose estimator returns exactly the target -> all errors ~0.
    m._estimate_poses = lambda sample, n: (traj.copy(), None)

    sample.quality_metrics = QualityMetrics()
    result = m.process(sample)
    assert result is sample
    qm = sample.quality_metrics
    assert qm.camera_rot_error == pytest.approx(0.0, abs=1e-6)
    assert qm.camera_trans_error == pytest.approx(0.0, abs=1e-6)
    assert qm.camera_traj_consistency == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Graceful unavailable: no real backend -> _backend "unavailable", fields None
# ---------------------------------------------------------------------------


def test_graceful_unavailable(monkeypatch, video_sample):
    import ayase.modules.camera_trajectory as ct

    # Force both backends absent: vggt is not installed (ImportError), and no
    # COLMAP/GLOMAP binary is discoverable.
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(
        "ayase.managed_native_tools.resolve_native_tool",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    m = ct.CameraTrajectoryModule()
    m.setup()
    assert m._backend == "unavailable"
    assert m._ml_available is False

    video_sample.quality_metrics = QualityMetrics()
    result = m.process(video_sample)
    assert result is video_sample
    assert video_sample.quality_metrics.camera_rot_error is None
    assert video_sample.quality_metrics.camera_trans_error is None
    assert video_sample.quality_metrics.camera_traj_consistency is None


def test_setup_provisions_colmap_and_glomap_in_separate_environments(monkeypatch):
    import ayase.modules.camera_trajectory as ct

    calls = []

    def fake_resolve(tool, packages, models_dir, **kwargs):
        calls.append((tool, packages))
        return f"managed/{tool}.exe"

    monkeypatch.setattr(ct.CameraTrajectoryModule, "_detect_sfm_binaries", lambda self: None)
    monkeypatch.setattr("ayase.managed_native_tools.resolve_native_tool", fake_resolve)

    module = ct.CameraTrajectoryModule({"pose_backend": "glomap"})
    module.setup()

    assert module._backend == "glomap"
    assert calls == [
        ("colmap", ("colmap=4.2.1=cpu_*",)),
        ("glomap", ("glomap=1.2.0=cpu_*",)),
    ]

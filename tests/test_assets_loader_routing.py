"""Loader-level contracts for configurable Ayase asset repositories."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, List, Tuple


def _install_fake_torchvision(monkeypatch: Any, builds: List[Any]) -> None:
    class FakeModel:
        def to(self, device: str) -> "FakeModel":
            self.device = device
            return self

        def eval(self) -> "FakeModel":
            return self

    class FakeWeights:
        def transforms(self) -> str:
            return "fake-transforms"

    optical_flow = ModuleType("torchvision.models.optical_flow")
    optical_flow.Raft_Large_Weights = SimpleNamespace(DEFAULT=FakeWeights())

    def fake_raft_large(*, weights: Any, progress: bool) -> FakeModel:
        builds.append((weights, progress))
        return FakeModel()

    optical_flow.raft_large = fake_raft_large
    models = ModuleType("torchvision.models")
    models.__path__ = []  # type: ignore[attr-defined]
    torchvision = ModuleType("torchvision")
    torchvision.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)
    monkeypatch.setitem(sys.modules, "torchvision.models.optical_flow", optical_flow)


def test_flow_routes_repo_id_and_url_and_keys_cache_by_resolved_source(
    monkeypatch: Any, tmp_path: Path
) -> None:
    from ayase import config as config_module
    from ayase import flow
    from ayase import runtime

    downloads: List[Tuple[str, str, str]] = []
    builds: List[Any] = []
    _install_fake_torchvision(monkeypatch, builds)
    monkeypatch.setattr(runtime, "resolve_torch_device", lambda _device: "cpu")
    monkeypatch.setattr(
        config_module,
        "download_torch_hub_checkpoint",
        lambda filename, url, models_dir: downloads.append((filename, url, str(models_dir))),
    )
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch-home"))
    flow._MODELS.clear()

    first = flow.load_raft_flow_model(
        models_dir=str(tmp_path), config={"assets_repo": "mirror/assets"}
    )
    equivalent = flow.load_raft_flow_model(
        models_dir=str(tmp_path),
        config={"assets_repo": "https://huggingface.co/mirror/assets"},
    )
    second_repo = flow.load_raft_flow_model(
        models_dir=str(tmp_path), config={"assets_repo": "other/assets"}
    )

    assert first == equivalent
    assert second_repo[2] == "cpu"
    assert len(builds) == 2
    assert [call[1] for call in downloads] == [
        "https://huggingface.co/mirror/assets/resolve/main/"
        "advanced_flow/raft_large_C_T_SKHT_V2-ff5fadd5.pth",
        "https://huggingface.co/other/assets/resolve/main/"
        "advanced_flow/raft_large_C_T_SKHT_V2-ff5fadd5.pth",
    ]


def test_pose_loader_forwards_config_to_both_asset_urls(monkeypatch: Any, tmp_path: Path) -> None:
    from ayase import config as config_module
    from ayase import pose
    from ayase import runtime

    downloads: List[Tuple[str, str, str]] = []

    class FakeBackend:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_rtmlib = ModuleType("rtmlib")
    fake_rtmlib.YOLOX = FakeBackend
    fake_rtmlib.RTMPose = FakeBackend
    monkeypatch.setitem(sys.modules, "rtmlib", fake_rtmlib)
    monkeypatch.setattr(runtime, "resolve_torch_device", lambda _device: "cpu")
    monkeypatch.setattr(
        config_module,
        "download_model_file",
        lambda relative, url, models_dir: downloads.append((relative, url, str(models_dir)))
        or (tmp_path / relative),
    )
    pose._DETPOSE_CACHE.clear()

    backend = pose.load_pose_backend(
        models_dir=str(tmp_path),
        config={"assets_repo": "https://huggingface.co/mirror/assets"},
    )

    assert backend is not None
    assert [call[1] for call in downloads] == [
        "https://huggingface.co/mirror/assets/resolve/main/" "rtmpose_fidelity/yolox_m.onnx",
        "https://huggingface.co/mirror/assets/resolve/main/" "rtmpose_fidelity/rtmpose_m.onnx",
    ]
    assert [call[0] for call in downloads] == [
        "rtmpose_fidelity/yolox_m.onnx",
        "rtmpose_fidelity/rtmpose_m.onnx",
    ]


def test_blendshape_loader_routes_full_repo_url_without_changing_cache_path(
    monkeypatch: Any, tmp_path: Path
) -> None:
    from ayase.modules import _blendshape_utils as blendshape

    model_path = tmp_path / blendshape.MODEL_FILENAME
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"face-landmarker")
    downloads: List[Tuple[str, str, str]] = []
    fake_mediapipe = ModuleType("mediapipe")
    fake_mediapipe.tasks = SimpleNamespace(
        vision=SimpleNamespace(
            FaceLandmarker=object(),
            FaceLandmarkerOptions=object(),
            RunningMode=object(),
        )
    )
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe)
    monkeypatch.setattr(
        blendshape,
        "download_model_file",
        lambda relative, url, models_dir: downloads.append((relative, url, str(models_dir)))
        or model_path,
    )

    extractor = blendshape.BlendshapeExtractor(
        str(tmp_path),
        config={"assets_repo": "https://huggingface.co/mirror/assets"},
    )

    assert extractor.setup()
    assert downloads == [
        (
            blendshape.MODEL_FILENAME,
            "https://huggingface.co/mirror/assets/resolve/"
            f"{blendshape.MODEL_REVISION}/{blendshape.MODEL_FILENAME}",
            str(tmp_path),
        )
    ]

"""Tests for dino_face_identity module."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from ..conftest import _test_module_basics


class _FakeDINO:
    """Minimal DINO model implementing the setup-time interface."""

    def load_state_dict(self, state):
        """Accept a fake mirrored state dictionary."""
        del state

    def eval(self):
        """Return the fake model after switching to evaluation mode."""
        return self

    def to(self, device):
        """Return the fake model after a device move."""
        del device
        return self


def _install_setup_fakes(monkeypatch, tmp_path):
    """Install lightweight dependency fakes and return captured loader calls."""
    hub_calls = []
    download_calls = []

    def hub_load(repo, model, **kwargs):
        hub_calls.append((repo, model, kwargs))
        return _FakeDINO()

    torch = ModuleType("torch")
    torch.hub = SimpleNamespace(load=hub_load)
    torch.load = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "torch", torch)

    transforms = SimpleNamespace(
        Compose=lambda steps: steps,
        Resize=lambda size: ("resize", size),
        ToTensor=lambda: "tensor",
        Normalize=lambda **kwargs: ("normalize", kwargs),
    )
    torchvision = ModuleType("torchvision")
    torchvision.transforms = transforms
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)

    class FakeFaceAnalysis:
        """Minimal InsightFace detector used by module setup."""

        def __init__(self, **kwargs):
            del kwargs

        def prepare(self, **kwargs):
            del kwargs

    insightface = ModuleType("insightface")
    insightface_app = ModuleType("insightface.app")
    insightface_app.FaceAnalysis = FakeFaceAnalysis
    insightface.app = insightface_app
    monkeypatch.setitem(sys.modules, "insightface", insightface)
    monkeypatch.setitem(sys.modules, "insightface.app", insightface_app)

    monkeypatch.setattr("ayase.runtime.resolve_torch_device", lambda _device: "cpu")

    def fake_download(*args):
        download_calls.append(args)
        return tmp_path / "weights.pth"

    monkeypatch.setattr("ayase.config.download_model_file", fake_download)
    return hub_calls, download_calls


def test_dino_face_identity_basics():
    from ayase.modules.dino_face_identity import DINOFaceIdentityModule

    _test_module_basics(DINOFaceIdentityModule, "dino_face_identity")


def test_dino_face_identity_image_without_setup(image_sample):
    from ayase.modules.dino_face_identity import DINOFaceIdentityModule

    module = DINOFaceIdentityModule()
    result = module.process(image_sample)
    assert result is image_sample


def test_dino_face_identity_video_without_setup(video_sample):
    from ayase.modules.dino_face_identity import DINOFaceIdentityModule

    module = DINOFaceIdentityModule()
    result = module.process(video_sample)
    assert result is video_sample


def test_default_repo_reference_is_preserved(monkeypatch, tmp_path):
    from ayase.modules.dino_face_identity import DINOFaceIdentityModule

    hub_calls, download_calls = _install_setup_fakes(monkeypatch, tmp_path)
    module = DINOFaceIdentityModule()
    module.setup()

    assert hub_calls == [("facebookresearch/dinov2", "dinov2_vitb14", {"pretrained": False})]
    assert download_calls == [
        (
            "dino_face_identity/dinov2_vitb14_pretrain.pth",
            "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/"
            "dino_face_identity/dinov2_vitb14_pretrain.pth",
            "models",
        )
    ]


def test_model_metadata_distinguishes_architecture_pin_from_default_weights():
    from ayase.modules.dino_face_identity import DINOFaceIdentityModule

    architecture, checkpoint = DINOFaceIdentityModule.models
    assert architecture["id"] == "facebookresearch/dinov2"
    assert architecture["type"] == "torch_hub"
    assert "architecture code only" in architecture["notes"]
    assert checkpoint == {
        "id": "dino_face_identity/dinov2_vitb14_pretrain.pth",
        "type": "local",
        "url": (
            "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/"
            "dino_face_identity/dinov2_vitb14_pretrain.pth"
        ),
        "task": "Default dinov2_vitb14 checkpoint weights",
        "notes": (
            "Downloaded from the mutable main revision; repo_revision does not pin "
            "this artifact."
        ),
    }


def test_pinned_repo_revision_is_used_for_mirrored_weights(monkeypatch, tmp_path):
    from ayase.modules.dino_face_identity import DINOFaceIdentityModule

    revision = "0123456789abcdef0123456789abcdef01234567"
    hub_calls, download_calls = _install_setup_fakes(monkeypatch, tmp_path)
    module = DINOFaceIdentityModule({"repo_revision": revision})
    module.setup()

    assert hub_calls == [
        (f"facebookresearch/dinov2:{revision}", "dinov2_vitb14", {"pretrained": False})
    ]
    assert len(download_calls) == 1


def test_pinned_repo_revision_is_used_for_upstream_weights(monkeypatch, tmp_path):
    from ayase.modules.dino_face_identity import DINOFaceIdentityModule

    revision = "fedcba9876543210fedcba9876543210fedcba98"
    hub_calls, download_calls = _install_setup_fakes(monkeypatch, tmp_path)
    module = DINOFaceIdentityModule({"model_name": "dinov2_vits14", "repo_revision": revision})
    module.setup()

    assert hub_calls == [(f"facebookresearch/dinov2:{revision}", "dinov2_vits14", {})]
    assert download_calls == []


@pytest.mark.parametrize("revision", ["main", "a" * 39, "g" * 40, 123])
def test_invalid_repo_revision_fails_without_loading(monkeypatch, tmp_path, revision):
    from ayase.modules.dino_face_identity import DINOFaceIdentityModule

    hub_calls, download_calls = _install_setup_fakes(monkeypatch, tmp_path)
    module = DINOFaceIdentityModule({"repo_revision": revision})
    module.setup()

    assert module._dino is None
    assert module._backend == "unavailable"
    assert hub_calls == []
    assert download_calls == []

"""Tests for the upstream VQA² module."""

import zipfile

import pytest

from tests.modules.conftest import _test_module_basics


def test_vqa2_basics():
    from ayase.modules.vqa2 import VQA2Module

    _test_module_basics(VQA2Module, "vqa2")
    assert VQA2Module.requires_external_backend is False


def test_vqa2_wa5_range_and_stability():
    from ayase.modules.vqa2 import _wa5

    assert _wa5([0.0] * 5) == pytest.approx(0.6)
    assert 0.99 < _wa5([1000, 0, 0, 0, 0]) <= 1.0
    assert 0.2 <= _wa5([0, 0, 0, 0, 1000]) < 0.21
    with pytest.raises(ValueError):
        _wa5([0.0] * 4)
    with pytest.raises(ValueError):
        _wa5([0.0, 0.0, float("nan"), 0.0, 0.0])


def test_vqa2_without_setup_is_graceful(image_sample):
    from ayase.modules.vqa2 import VQA2Module

    result = VQA2Module().process(image_sample)
    assert result is image_sample
    assert result.quality_metrics is None


def test_vqa2_processes_image_score(image_sample, monkeypatch):
    from ayase.modules.vqa2 import VQA2Module

    module = VQA2Module()
    module._backend = "vqa2"
    module._model = object()
    monkeypatch.setattr(module, "_load_image", lambda _path: object())
    monkeypatch.setattr(module, "_score_image", lambda _image: 0.82)

    result = module.process(image_sample)

    assert result is image_sample
    assert result.quality_metrics.vqa2_score == pytest.approx(0.82)


def test_vqa2_processes_video_score(video_sample, monkeypatch):
    from ayase.modules.vqa2 import VQA2Module

    module = VQA2Module()
    module._backend = "vqa2"
    module._model = object()
    monkeypatch.setattr(module, "_load_video", lambda _path: ([object()] * 4, [object()]))
    monkeypatch.setattr(module, "_score_video", lambda _slow, _fast: 0.71)

    result = module.process(video_sample)

    assert result is video_sample
    assert result.quality_metrics.vqa2_score == pytest.approx(0.71)


def test_vqa2_rejects_out_of_range_score(image_sample, monkeypatch):
    from ayase.modules.vqa2 import VQA2Module

    module = VQA2Module()
    module._backend = "vqa2"
    module._model = object()
    monkeypatch.setattr(module, "_load_image", lambda _path: object())
    monkeypatch.setattr(module, "_score_image", lambda _image: 1.1)

    result = module.process(image_sample)

    assert result is image_sample
    assert result.quality_metrics is None


def test_vqa2_extracts_pinned_runtime_and_patches_slowfast(tmp_path):
    from ayase.modules.vqa2 import VQA2Module, VQA2_SOURCE_REVISION

    prefix = (
        "Visual-Question-Answering-for-Video-Quality-Assessment-"
        f"{VQA2_SOURCE_REVISION}/quality_scoring/"
    )
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(prefix + "llava/__init__.py", "")
        bundle.writestr(
            prefix + "llava/model/slowfast/builder.py",
            "import torch\nmodel = torch.load('slowfast.pth',weights_only=False)\n",
        )
        bundle.writestr(
            prefix + "llava/model/llava_arch.py",
            "size = image_sizes[NUM]\n",
        )
        bundle.writestr(prefix + "LICENSE", "Apache License 2.0")
        bundle.writestr(prefix + "ignored.bin", "not extracted")

    runtime = VQA2Module._extract_runtime(
        archive, tmp_path / "runtime", VQA2_SOURCE_REVISION
    )

    builder = runtime / "llava" / "model" / "slowfast" / "builder.py"
    source = builder.read_text(encoding="utf-8")
    assert "AYASE_VQA2_SLOWFAST_PATH" in source
    assert "slowfast.pth',weights_only=False" not in source
    architecture = runtime / "llava" / "model" / "llava_arch.py"
    assert "image_sizes[image_idx]" in architecture.read_text(encoding="utf-8")
    assert not (runtime / "ignored.bin").exists()


def test_vqa2_runtime_archive_rejects_path_traversal(tmp_path):
    from ayase.modules.vqa2 import VQA2Module, VQA2_SOURCE_REVISION

    prefix = (
        "Visual-Question-Answering-for-Video-Quality-Assessment-"
        f"{VQA2_SOURCE_REVISION}/quality_scoring/"
    )
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(prefix + "llava/__init__.py", "")
        bundle.writestr(prefix + "llava/../../escape.py", "bad")

    with pytest.raises(ValueError, match="Unsafe VQA² archive member"):
        VQA2Module._extract_runtime(
            archive, tmp_path / "runtime", VQA2_SOURCE_REVISION
        )


def test_vqa2_declares_all_reference_assets():
    from ayase.modules.vqa2 import (
        VQA2_MODEL_ID,
        VQA2_SLOWFAST_ID,
        VQA2Module,
    )

    model_ids = {entry["id"] for entry in VQA2Module.models}
    assert VQA2_MODEL_ID in model_ids
    assert VQA2_SLOWFAST_ID in model_ids
    assert any(model_id.startswith("VQA2-source-") for model_id in model_ids)


def test_vqa2_runtime_dependencies_are_declared():
    from pathlib import Path

    project = Path(__file__).resolve().parents[3] / "pyproject.toml"
    pyproject = project.read_text(encoding="utf-8")

    assert '"einops-exts>=0.0.4,<0.1"' in pyproject
    assert '"pytorchvideo>=0.1.5,<0.2"' in pyproject
    assert '"decord>=0.6.0"' in pyproject

"""Tests for the upstream VQA² module."""


from pathlib import Path

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


def test_vqa2_runtime_is_vendored_and_patched():
    """The scorer runs on the in-tree copy: nothing is fetched at setup time."""
    from ayase.modules.vqa2 import VQA2Module

    runtime = VQA2Module._vendored_runtime()

    assert (runtime / "llava" / "__init__.py").is_file()
    builder = runtime / "llava" / "model" / "slowfast" / "builder.py"
    source = builder.read_text(encoding="utf-8")
    assert "AYASE_VQA2_SLOWFAST_PATH" in source
    assert "slowfast.pth',weights_only=False" not in source
    architecture = runtime / "llava" / "model" / "llava_arch.py"
    assert "image_sizes[image_idx]" in architecture.read_text(encoding="utf-8")


def test_vqa2_runtime_lives_inside_the_package():
    """A path outside the installed package would break a wheel install."""
    import ayase
    from ayase.modules.vqa2 import VQA2Module

    package_root = Path(ayase.__file__).resolve().parent
    assert VQA2Module._vendored_runtime().is_relative_to(package_root)


def test_vqa2_downloads_no_source_code():
    """Weights are fetched, code is not: no source archive may be declared."""
    from ayase.modules.vqa2 import VQA2Module

    for entry in VQA2Module.models:
        url = str(entry.get("url", ""))
        assert not url.endswith(".zip"), entry["id"]


def test_vqa2_declares_all_reference_assets():
    from ayase.modules.vqa2 import (
        VQA2_MODEL_ID,
        VQA2_SLOWFAST_ID,
        VQA2Module,
    )

    model_ids = {entry["id"] for entry in VQA2Module.models}
    assert VQA2_MODEL_ID in model_ids
    assert VQA2_SLOWFAST_ID in model_ids
    assert not any(model_id.startswith("VQA2-source-") for model_id in model_ids)


def test_vqa2_runtime_dependencies_are_declared():
    from pathlib import Path

    project = Path(__file__).resolve().parents[3] / "pyproject.toml"
    pyproject = project.read_text(encoding="utf-8")

    assert '"einops-exts>=0.0.4,<0.1"' in pyproject
    assert '"pytorchvideo>=0.1.5,<0.2"' in pyproject
    assert '"decord>=0.6.0"' in pyproject

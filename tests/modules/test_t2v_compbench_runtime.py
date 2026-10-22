"""Tests for the self-managed T2V-CompBench runtime."""

import zipfile
from pathlib import Path


def test_source_download_is_secure_verified_and_cached(tmp_path, monkeypatch):
    from ayase.modules import _t2v_compbench_runtime as runtime

    payload = b"official evaluator\n"
    fixture = tmp_path / "fixture.zip"
    with zipfile.ZipFile(fixture, "w") as archive:
        archive.writestr("snapshot/evaluator.py", payload)
        archive.writestr(
            "snapshot/Grounded-Segment-Anything/GroundingDINO/groundingdino/models/"
            "GroundingDINO/ms_deform_attn.py",
            "try:\n    from groundingdino import _C\nexcept:\n    pass\nif value.is_cuda:\n    pass\n",
        )
    source = tmp_path / "payload"
    source.write_bytes(payload)
    monkeypatch.setattr(runtime, "PINNED_BLOBS", {"evaluator.py": runtime.git_blob_hash(source)})
    calls = []

    def download(_url, destination):
        calls.append(destination)
        destination.write_bytes(fixture.read_bytes())

    first = runtime.ensure_source(tmp_path / "models", downloader=download)
    second = runtime.ensure_source(tmp_path / "models", downloader=download)
    assert first == second
    assert len(calls) == 1
    patched = first / (
        "Grounded-Segment-Anything/GroundingDINO/groundingdino/models/"
        "GroundingDINO/ms_deform_attn.py"
    )
    assert "value.is_cuda and _C is not None" in patched.read_text(encoding="utf-8")


def test_archive_rejects_path_traversal(tmp_path):
    import pytest
    from ayase.modules import _t2v_compbench_runtime as runtime

    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("snapshot/../../escape.py", "bad")
    with pytest.raises(ValueError, match="unsafe"):
        runtime._safe_extract(archive, tmp_path / "output")


def test_environment_groups_are_private_and_source_fingerprinted(tmp_path):
    from ayase.modules import _t2v_compbench_runtime as runtime

    root = tmp_path / "source"
    (root / "LLaVA").mkdir(parents=True)
    (root / "Grounded-Segment-Anything/segment_anything").mkdir(parents=True)
    (root / "Depth-Anything").mkdir()
    (root / "dot").mkdir()
    calls = []

    def provision(name, requirements, models_dir, **kwargs):
        calls.append((name, requirements, Path(models_dir), kwargs))
        return tmp_path / name / "python"

    environments = runtime.ensure_environments(
        root,
        ["consistent_attribute_binding", "motion_binding"],
        tmp_path / "models",
        provisioner=provision,
    )
    assert set(environments) == {"llava", "vision"}
    assert calls[0][0] == "t2v-compbench-llava"
    assert root / "LLaVA" in calls[0][1]
    assert calls[0][3]["python_version"] == "3.10.15"
    assert calls[0][3]["pip_options"] == runtime.CUDA_PIP_OPTIONS
    assert calls[1][0] == "t2v-compbench-vision"
    assert not any("git+" in str(item) for item in calls[1][1])
    assert root / "Grounded-Segment-Anything" in calls[1][3]["source_roots"]
    assert calls[1][3]["python_version"] == "3.12.3"
    assert calls[1][3]["pip_options"] == runtime.CUDA_PIP_OPTIONS


def test_assets_download_only_for_selected_categories(tmp_path):
    from ayase.modules import _t2v_compbench_runtime as runtime

    downloaded = []

    def download(url, destination):
        downloaded.append((url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"checkpoint")

    runtime.ensure_assets(tmp_path, ["generative_numeracy"], downloader=download)
    assert [path.name for _url, path in downloaded] == ["groundingdino_swint_ogc.pth"]


def test_gpu_probe_failure_is_a_clean_unavailable(monkeypatch):
    from ayase.modules import _t2v_compbench_runtime as runtime

    class _Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    monkeypatch.setitem(__import__("sys").modules, "torch", _Torch)
    monkeypatch.setattr(runtime.shutil, "which", lambda _name: "nvidia-smi")
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("probe failed")),
    )
    assert runtime.gpu_is_available() is False

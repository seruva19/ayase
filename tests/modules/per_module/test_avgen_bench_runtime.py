"""Portable source-cache and staging contracts, without model or API access."""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from ayase.third_party.avgen_bench import runtime


def _archive(tmp_path, *, name="eval/head.py", data=b"source\n", symlink=False):
    archive = tmp_path / "source.zip"
    member = zipfile.ZipInfo("root/" + name)
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK if symlink else stat.S_IFREG) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, data)
    manifest = {
        "revision": runtime.SOURCE_REVISION,
        "archive_root": "root",
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "members": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}},
    }
    return archive, manifest


def test_source_cache_is_automatically_acquired_and_revalidated(tmp_path, monkeypatch):
    archive, manifest = _archive(tmp_path)
    calls = []
    monkeypatch.setattr(runtime, "_source_manifest", lambda: manifest)

    def download(expected, target):
        calls.append(expected["revision"])
        shutil.copyfile(archive, target)

    monkeypatch.setattr(runtime, "_download_source_archive", download)
    source = runtime.ensure_source(tmp_path / "models")
    assert (source / "eval/head.py").read_bytes() == b"source\n"
    assert runtime.ensure_source(tmp_path / "models") == source
    assert len(calls) == 1
    (source / "eval/head.py").write_bytes(b"altered")
    with pytest.raises(ValueError, match="verification failed"):
        runtime.ensure_source(tmp_path / "models")


def test_test_mode_stops_before_manifest_or_filesystem_access(tmp_path, monkeypatch):
    def unexpected():
        pytest.fail("source manifest must not be opened in test mode")

    monkeypatch.setattr(runtime, "_source_manifest", unexpected)
    with pytest.raises(RuntimeError, match="test mode"):
        runtime.ensure_source(tmp_path / "models", test_mode=True)
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("name", ["../escape.py", "/absolute.py", "bad\\path.py", "bad:stream.py"])
def test_source_extraction_rejects_unsafe_selected_paths(tmp_path, name):
    archive, manifest = _archive(tmp_path, name=name)
    with pytest.raises(ValueError, match="unsafe"):
        runtime._extract_source(archive, tmp_path / "output", manifest)
    assert not (tmp_path / "escape.py").exists()


def test_source_extraction_rejects_selected_symlinks(tmp_path):
    archive, manifest = _archive(tmp_path, symlink=True)
    with pytest.raises(ValueError, match="regular file"):
        runtime._extract_source(archive, tmp_path / "output", manifest)


def test_source_extraction_verifies_payload_bytes(tmp_path):
    archive, manifest = _archive(tmp_path)
    manifest["members"]["eval/head.py"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="verification failed"):
        runtime._extract_source(archive, tmp_path / "output", manifest)


def _record(tmp_path):
    original = tmp_path / "video with spaces 雪.mp4"
    original.write_bytes(b"original media fixture")
    return {
        "id": "sample-1",
        "video_path": str(original),
        "category": "ads",
        "content": 'Title: "snow" | 家',
        "prompt_text": "Original prompt\n雪 and sound.",
        "expected_faces": {"primary_faces": {"min_ids": 1, "max_ids": 2}},
        "mode": "custom",
    }


@pytest.mark.parametrize("force_copy", [False, True])
def test_staging_preserves_media_prompts_and_original_identity(tmp_path, monkeypatch, force_copy):
    row = _record(tmp_path)
    if force_copy:

        def unavailable_link(*args):
            raise OSError("cross-device link unavailable")

        monkeypatch.setattr(runtime.os, "link", unavailable_link)
    staged = runtime.stage_records([row], tmp_path / "run")
    record = staged["records"][0]
    token = record["safe_content"]
    assert not any(character in token for character in ':"|/\\')
    assert record["content"] == row["content"]
    assert record["prompt_text"] == row["prompt_text"]
    assert Path(record["staged_video_path"]).read_bytes() == Path(row["video_path"]).read_bytes()
    assert record["staging_method"] == ("copy" if force_copy else "hardlink")
    prompts = json.loads((Path(staged["prompts_dir"]) / "ads.json").read_text(encoding="utf-8"))
    annotations = json.loads(
        (Path(staged["face_prompts_dir"]) / "ads.expected_faces.json").read_text(encoding="utf-8")
    )
    assert prompts == [{"content": token, "prompt": row["prompt_text"]}]
    assert annotations[0]["content"] == token
    assert annotations[0]["prompt"] == row["prompt_text"]
    assert annotations[0]["primary_faces"] == row["expected_faces"]["primary_faces"]
    assert "content" not in row["expected_faces"]


def test_staging_does_not_invent_face_annotations(tmp_path):
    row = _record(tmp_path)
    row.pop("expected_faces")
    staged = runtime.stage_records([row], tmp_path / "run")
    assert list(Path(staged["face_prompts_dir"]).iterdir()) == []


@pytest.mark.parametrize("category", ["../ads", "Ads", "nul", "news/ads"])
def test_staging_rejects_ambiguous_or_unsafe_categories(tmp_path, category):
    row = _record(tmp_path)
    row["category"] = category
    with pytest.raises(ValueError):
        runtime.stage_records([row], tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_staging_does_not_relabel_non_mp4_media(tmp_path):
    row = _record(tmp_path)
    source = tmp_path / "audio-video.mkv"
    source.write_bytes(b"a different container")
    row["video_path"] = str(source)
    with pytest.raises(ValueError, match="requires MP4"):
        runtime.stage_records([row], tmp_path / "run")


def test_staging_rejects_duplicate_identities_before_copying(tmp_path):
    row = _record(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        runtime.stage_records([row, {**row, "id": "sample-2"}], tmp_path / "run")
    assert not (tmp_path / "run").exists()


def _worker_request(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    return {
        "schema": 1,
        "source_revision": runtime.SOURCE_REVISION,
        "dimension": "Holistic",
        "source_dir": str(source),
        "work_dir": str(tmp_path / "work"),
        "models_dir": str(tmp_path / "models"),
        "output_dir": str(tmp_path / "outputs"),
        "config": {},
        "assets": {},
        "records": [],
    }


def test_host_executes_real_standalone_worker_without_persisting_credentials(tmp_path):
    request = _worker_request(tmp_path)
    result = runtime.execute_worker(
        request, sys.executable, api_key="synthetic-private-value", timeout_s=30
    )
    assert result["records"] == []
    assert result["counts"]["scored"] == 0
    assert result["dimension"] == "Holistic"
    for path in (tmp_path / "work").iterdir():
        assert b"synthetic-private-value" not in path.read_bytes()


def test_host_rejects_credentials_in_request_files(tmp_path):
    request = _worker_request(tmp_path)
    request["config"]["gemini_api_key"] = "synthetic-private-value"
    with pytest.raises(ValueError, match="credentials privately"):
        runtime.execute_worker(request, sys.executable)
    assert not (tmp_path / "work").exists()


def test_host_test_mode_never_launches_worker(tmp_path, monkeypatch):
    request = _worker_request(tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("test mode must not start a process")

    monkeypatch.setattr(runtime.subprocess, "Popen", unexpected)
    with pytest.raises(RuntimeError, match="test mode"):
        runtime.execute_worker(request, sys.executable, test_mode=True)
    assert not (tmp_path / "work").exists()


def test_worker_timeout_reaps_the_created_process(tmp_path, monkeypatch):
    request = _worker_request(tmp_path)
    sleeper = tmp_path / "sleep.py"
    sleeper.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "_RUNNER_PATH", sleeper)
    original_popen = runtime.subprocess.Popen
    created = []

    def capture(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", capture)
    with pytest.raises(subprocess.TimeoutExpired):
        runtime.execute_worker(request, sys.executable, timeout_s=0.2)
    assert created
    assert all(process.poll() is not None for process in created)


@pytest.mark.parametrize("system", ["nt", "posix"])
@pytest.mark.parametrize("conda", [False, True])
def test_worker_launch_discovers_private_gpu_libraries_for_conda_and_venv(
    tmp_path, monkeypatch, system, conda
):
    prefix = tmp_path / "private"
    if system == "nt":
        python = prefix / ("python.exe" if conda else "Scripts/python.exe")
        site = prefix / "Lib/site-packages"
        platform_libraries = [
            site / "torch/lib",
            site / "nvidia/cublas/bin",
            site / "nvidia/cudnn/lib",
        ]
        conda_libraries = [prefix / "Library/bin"] if conda else []
        variable = "PATH"
        separator = ";"
    else:
        python = prefix / "bin/python"
        site = prefix / "lib/python3.10/site-packages"
        platform_libraries = [
            site / "torch/lib",
            site / "nvidia/cublas/lib",
            site / "nvidia/cudnn/bin",
        ]
        conda_libraries = [prefix / "lib"] if conda else []
        variable = "LD_LIBRARY_PATH"
        separator = ":"
    python.parent.mkdir(parents=True)
    python.touch()
    for directory in [*conda_libraries, *platform_libraries]:
        directory.mkdir(parents=True, exist_ok=True)
    if conda:
        (prefix / "conda-meta").mkdir()

    captured = {}

    class Child:
        def __init__(self, command, **kwargs):
            captured.update(kwargs["env"])
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "source_revision": runtime.SOURCE_REVISION,
                        "dimension": "Holistic",
                        "records": [],
                    }
                ),
                encoding="utf-8",
            )

        def wait(self, timeout):
            return 0

    environment = dict(os.environ)
    environment["PATH"] = "host-path"
    environment["LD_LIBRARY_PATH"] = "host-library-path"
    fake_os = SimpleNamespace(name=system, pathsep=separator, environ=environment)
    monkeypatch.setattr(runtime, "os", fake_os)
    monkeypatch.setattr(runtime.subprocess, "Popen", Child)

    runtime.execute_worker(_worker_request(tmp_path), python)

    configured = captured[variable]
    required = [*conda_libraries, *platform_libraries]
    assert all(str(directory) in configured for directory in required)
    existing = "host-path" if system == "nt" else "host-library-path"
    assert configured.endswith(existing)
    assert all(configured.index(str(directory)) < configured.index(existing) for directory in required)
    if conda:
        assert captured["CONDA_PREFIX"] == str(prefix)


def test_posix_worker_keeps_symlinked_venv_interpreter_and_its_library_prefix(
    tmp_path, monkeypatch
):
    base = tmp_path / "managed base"
    base_python = base / "bin/python"
    base_python.parent.mkdir(parents=True)
    base_python.touch()
    prefix = tmp_path / "private venv"
    python = prefix / "bin/python"
    python.parent.mkdir(parents=True)
    try:
        python.symlink_to(base_python)
    except OSError as exc:
        pytest.skip(f"file symlink creation is unavailable: {exc}")
    (prefix / "pyvenv.cfg").write_text(f"home = {base}\n", encoding="utf-8")
    private_library = prefix / "lib/python3.10/site-packages/nvidia/cublas/lib"
    private_library.mkdir(parents=True)
    captured = {}

    class Child:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["environment"] = kwargs["env"]
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "source_revision": runtime.SOURCE_REVISION,
                        "dimension": "Holistic",
                        "records": [],
                    }
                ),
                encoding="utf-8",
            )

        def wait(self, timeout):
            return 0

    fake_os = SimpleNamespace(name="posix", pathsep=":", environ=dict(os.environ))
    monkeypatch.setattr(runtime, "os", fake_os)
    monkeypatch.setattr(runtime.subprocess, "Popen", Child)

    runtime.execute_worker(_worker_request(tmp_path), python)

    launched = Path(captured["command"][0])
    assert launched == python.absolute()
    assert launched.is_symlink()
    assert launched.resolve() == base_python.resolve()
    assert str(private_library) in captured["environment"]["LD_LIBRARY_PATH"]

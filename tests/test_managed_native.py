"""Tests for private native-binary runtime provisioning."""

import hashlib
import io
import json
import tarfile
from pathlib import Path


def _archive(member: str, payload: bytes = b"binary") -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:bz2") as bundle:
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def test_micromamba_download_is_hashed_and_cached(monkeypatch, tmp_path):
    import ayase.managed_native as native

    member = "Library/bin/micromamba.exe"
    archive = _archive(member)
    monkeypatch.setattr(native, "_platform_subdir", lambda: "win-64")
    monkeypatch.setitem(
        native._MICROMAMBA_BUILDS,
        "win-64",
        ("https://example.invalid/micromamba.tar.bz2", hashlib.sha256(archive).hexdigest()),
    )
    calls = []

    def download(url, destination):
        calls.append(url)
        destination.write_bytes(archive)

    monkeypatch.setattr(native, "_download", download)
    first = native._micromamba(tmp_path)
    second = native._micromamba(tmp_path)
    assert first == second and first.read_bytes() == b"binary"
    assert calls == ["https://example.invalid/micromamba.tar.bz2"]


def test_micromamba_rejects_wrong_digest(monkeypatch, tmp_path):
    import pytest
    import ayase.managed_native as native

    monkeypatch.setattr(native, "_platform_subdir", lambda: "linux-64")
    monkeypatch.setitem(native._MICROMAMBA_BUILDS, "linux-64", ("https://invalid", "0" * 64))
    monkeypatch.setattr(
        native, "_download", lambda url, destination: destination.write_bytes(b"bad")
    )
    with pytest.raises(RuntimeError, match="SHA-256"):
        native._micromamba(tmp_path)


def test_environment_creation_uses_conda_forge_and_reuses_manifest(monkeypatch, tmp_path):
    import subprocess
    import ayase.managed_native as native

    tool = tmp_path / "micromamba.exe"
    tool.touch()
    monkeypatch.setattr(native, "_platform_subdir", lambda: "win-64")
    monkeypatch.setattr(native, "_micromamba", lambda models_dir: tool)
    monkeypatch.setattr(native, "_python", lambda environment: environment / "python.exe")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        prefix = Path(command[command.index("--prefix") + 1])
        prefix.mkdir(parents=True, exist_ok=True)
        (prefix / "python.exe").touch()
        kwargs["stdout"].write("created")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(native.subprocess, "run", run)
    first = native.ensure_managed_native_environment(
        "audio-native", ("pesq=0.0.4", "pyworld=0.3.5"), tmp_path
    )
    second = native.ensure_managed_native_environment(
        "audio-native", ("pesq=0.0.4", "pyworld=0.3.5"), tmp_path
    )
    assert first == second and len(calls) == 1
    command = calls[0][0]
    assert command[1] == "create"
    assert "--no-rc" in command and "--root-prefix" in command
    assert command[command.index("--channel") + 1] == "conda-forge"
    assert calls[0][1]["env"]["MAMBA_ROOT_PREFIX"].startswith(str(tmp_path))
    assert calls[0][1]["env"]["CONDA_PKGS_DIRS"].startswith(str(tmp_path))
    assert "python=3.11" in command and "pesq=0.0.4" in command
    manifest = json.loads((first.parent / "ayase-native.json").read_text(encoding="utf-8"))
    assert manifest["platform"] == "win-64"


def test_pip_requirements_install_into_native_prefix(monkeypatch, tmp_path):
    import subprocess
    import ayase.managed_native as native

    tool = tmp_path / "micromamba.exe"
    tool.touch()
    monkeypatch.setattr(native, "_platform_subdir", lambda: "win-64")
    monkeypatch.setattr(native, "_micromamba", lambda models_dir: tool)
    monkeypatch.setattr(native, "_python", lambda environment: environment / "python.exe")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "--prefix" in command:
            prefix = Path(command[command.index("--prefix") + 1])
            prefix.mkdir(parents=True)
            (prefix / "python.exe").touch()
        kwargs["stdout"].write("ok")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(native.subprocess, "run", run)
    native.ensure_managed_native_environment(
        "mixed",
        ("pyworld=0.3.5",),
        tmp_path,
        pip_requirements=("pysptk==1.0.1",),
    )
    assert calls[1][1:4] == ["-m", "pip", "install"]
    assert calls[1][-1] == "pysptk==1.0.1"


def test_failed_creation_is_not_cached(monkeypatch, tmp_path):
    import subprocess
    import pytest
    import ayase.managed_native as native

    tool = tmp_path / "micromamba"
    tool.touch()
    monkeypatch.setattr(native, "_platform_subdir", lambda: "linux-64")
    monkeypatch.setattr(native, "_micromamba", lambda models_dir: tool)

    def fail(command, **kwargs):
        kwargs["stdout"].write("solver failed")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(native.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="solver failed"):
        native.ensure_managed_native_environment("broken", ("pesq=0.0.4",), tmp_path)
    assert not list((tmp_path / ".ayase" / "environments").glob("native-broken-*"))


def test_resolves_executable_inside_private_prefix(monkeypatch, tmp_path):
    import ayase.managed_native as native

    monkeypatch.setattr(native.os, "name", "nt")
    python = tmp_path / "python.exe"
    python.touch()
    ffmpeg = tmp_path / "Library" / "bin" / "ffmpeg.exe"
    ffmpeg.parent.mkdir(parents=True)
    ffmpeg.touch()
    assert native.managed_native_executable(python, "ffmpeg") == ffmpeg

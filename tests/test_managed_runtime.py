"""Tests for isolated metric-runtime provisioning."""

import json
import os
import sys
from pathlib import Path

import pytest

from ayase import managed_runtime


def test_empty_environment_is_created_and_reused(tmp_path: Path) -> None:
    first = managed_runtime.ensure_managed_environment("empty", [], tmp_path)
    second = managed_runtime.ensure_managed_environment("empty", [], tmp_path)

    assert first == second
    assert first.is_file()
    manifest = json.loads((first.parent.parent / "ayase-managed.json").read_text(encoding="utf-8"))
    assert manifest["requirements"] == []
    assert manifest["fingerprint"]


def test_failed_install_is_logged_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installs = 0

    def fake_install(command, log_path):
        nonlocal installs
        installs += 1
        log_path.write_text("synthetic install failure", encoding="utf-8")
        return 1

    monkeypatch.setattr(managed_runtime, "_run_logged", fake_install)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="synthetic install failure"):
            managed_runtime.ensure_managed_environment("broken", ["missing==1"], tmp_path)

    assert installs == 2
    logs = list((tmp_path / ".ayase" / "logs").glob("broken-*.log"))
    assert len(logs) == 1
    assert "synthetic install failure" in logs[0].read_text(encoding="utf-8")
    assert not list((tmp_path / ".ayase" / "environments").glob("broken-*"))


def test_source_content_changes_fingerprint(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    file = source / "metric.py"
    file.write_text("VALUE = 1\n", encoding="utf-8")
    first = managed_runtime.ensure_managed_environment(
        "source", [], tmp_path / "models", source_roots=[source]
    )

    file.write_text("VALUE = 2\n", encoding="utf-8")
    second = managed_runtime.ensure_managed_environment(
        "source", [], tmp_path / "models", source_roots=[source]
    )

    assert first != second


def test_source_identity_ignores_model_weights(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    code = source / "metric.py"
    code.write_text("VALUE = 1\n", encoding="utf-8")
    before = managed_runtime._source_identity(source)

    (source / "model.safetensors").write_bytes(b"large model placeholder")
    assert managed_runtime._source_identity(source) == before

    code.write_text("VALUE = 2\n", encoding="utf-8")
    assert managed_runtime._source_identity(source) != before


def test_source_identity_does_not_require_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "metric.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        managed_runtime,
        "_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("git")),
    )

    identity = managed_runtime._source_identity(source)
    assert identity["kind"] == "content"
    assert identity["content"]


def test_pip_invocation_uses_managed_interpreter_and_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = []

    def fake_install(command, log_path):
        observed.append(command)
        log_path.write_text("installed", encoding="utf-8")
        return 0

    monkeypatch.setattr(managed_runtime, "_run_logged", fake_install)
    interpreter = managed_runtime.ensure_managed_environment(
        "options", ["demo==1.0"], tmp_path, pip_options=("--no-index",)
    )

    assert len(observed) == 1
    command = observed[0]
    assert Path(command[0]).name == ("python.exe" if os.name == "nt" else "python")
    assert command[1:] == ["-m", "pip", "install", "--no-index", "demo==1.0"]
    assert Path(command[0]) != Path(sys.executable)


def test_no_deps_stage_is_fingerprinted_logged_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = []
    fail_stage = True

    def fake_install(command, log_path):
        observed.append(list(command))
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(" ".join(command) + "\n")
        return 1 if fail_stage and "--no-deps" in command else 0

    monkeypatch.setattr(managed_runtime, "_run_logged", fake_install)
    with pytest.raises(RuntimeError, match="no-deps installation failed"):
        managed_runtime.ensure_managed_environment(
            "staged",
            ["runtime==1"],
            tmp_path,
            no_deps_requirements=["broken-metadata-package==2"],
        )

    fail_stage = False
    interpreter = managed_runtime.ensure_managed_environment(
        "staged",
        ["runtime==1"],
        tmp_path,
        no_deps_requirements=["broken-metadata-package==2"],
    )
    assert interpreter.is_file()
    assert len(observed) == 4
    assert observed[-1][-2:] == ["--no-deps", "broken-metadata-package==2"]
    manifest = json.loads((interpreter.parent.parent / "ayase-managed.json").read_text())
    assert manifest["no_deps_requirements"] == ["broken-metadata-package==2"]


def test_python_version_uses_acquired_interpreter_for_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "managed-python.exe"
    real_run = managed_runtime._run
    creation_commands = []

    monkeypatch.setattr(managed_runtime, "_resolve_managed_python", lambda version, root: selected)
    monkeypatch.setattr(
        managed_runtime,
        "_python_identity",
        lambda path: {"version": "3.10.15", "executable": str(path)},
    )

    def fake_run(command, *, cwd=None, env=None):
        if command[0] == str(selected) and command[1:3] == ["-m", "venv"]:
            creation_commands.append(command)
            command = [sys.executable, *command[1:]]
        return real_run(command, cwd=cwd, env=env)

    monkeypatch.setattr(managed_runtime, "_run", fake_run)
    interpreter = managed_runtime.ensure_managed_environment(
        "alternate", [], tmp_path / "models", python_version="3.10.15"
    )

    assert creation_commands
    assert creation_commands[0][0] == str(selected)
    manifest = json.loads((interpreter.parent.parent / "ayase-managed.json").read_text())
    assert manifest["python_request"] == "3.10.15"


def test_python_and_version_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        managed_runtime.ensure_managed_environment(
            "invalid", [], tmp_path, python=sys.executable, python_version="3.10"
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific process probe")
def test_windows_liveness_probe_does_not_use_os_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        managed_runtime.os,
        "kill",
        lambda *args: pytest.fail("os.kill must not be used for Windows liveness checks"),
    )
    assert managed_runtime._process_is_alive(os.getpid()) is True
    assert managed_runtime._process_is_alive(0x7FFFFFFF) is False

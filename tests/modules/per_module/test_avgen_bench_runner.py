"""Run the real standalone IPC entrypoint with Python site packages disabled."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ayase.third_party.avgen_bench import runtime
from ayase.third_party.avgen_bench import _native_runner


def test_boolean_ipc_version_is_not_an_integer_version():
    with pytest.raises(ValueError, match="schema/source revision"):
        _native_runner.execute(
            {"schema": True, "source_revision": runtime.SOURCE_REVISION, "dimension": "Vis"}
        )


def test_worker_error_redaction(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-private-token")
    assert (
        _native_runner._redact("failed synthetic-private-token request")
        == "failed [redacted] request"
    )


@pytest.mark.parametrize("conda", [False, True])
def test_windows_runner_keeps_private_gpu_dll_directories_alive(tmp_path, monkeypatch, conda):
    prefix = tmp_path / "private"
    site = prefix / "Lib/site-packages"
    expected = [
        site / "torch/lib",
        site / "nvidia/cublas/bin",
        site / "nvidia/cudnn/lib",
    ]
    if conda:
        (prefix / "conda-meta").mkdir(parents=True)
        expected.insert(0, prefix / "Library/bin")
    for directory in expected:
        directory.mkdir(parents=True, exist_ok=True)

    seen = []
    handles = []

    def add_dll_directory(path):
        seen.append(path)
        return "handle:" + path

    monkeypatch.setattr(
        _native_runner,
        "os",
        SimpleNamespace(name="nt", add_dll_directory=add_dll_directory),
    )
    monkeypatch.setattr(_native_runner.sys, "prefix", str(prefix))
    monkeypatch.setattr(_native_runner, "_DLL_HANDLES", handles)

    _native_runner._configure_native_libraries()

    assert seen == [str(directory) for directory in expected]
    assert handles == ["handle:" + str(directory) for directory in expected]


@pytest.mark.parametrize(
    "dimension",
    ["Vis", "Aud", "AV", "Lip", "Face", "Lo-Phy", "Text", "Music", "Speech", "Hi-Phy", "Holistic"],
)
def test_empty_dimension_dispatch_has_no_host_or_model_dependencies(tmp_path, dimension):
    runner = Path(runtime.__file__).with_name("_native_runner.py")
    source = tmp_path / "source with spaces"
    source.mkdir()
    request = {
        "schema": 1,
        "source_revision": runtime.SOURCE_REVISION,
        "dimension": dimension,
        "source_dir": str(source),
        "models_dir": str(tmp_path / "models"),
        "work_dir": str(tmp_path / "work"),
        "output_dir": str(tmp_path / "output"),
        "records": [],
        "config": {},
        "assets": {},
    }
    input_file, result_file = tmp_path / "input.json", tmp_path / "result.json"
    input_file.write_text(json.dumps(request), encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(runner),
            "--request",
            str(input_file),
            "--result",
            str(result_file),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    document = json.loads(result_file.read_text(encoding="utf-8"))
    assert document.get("error") is None, document
    assert document["dimension"] == dimension
    assert document["records"] == []
    assert document["counts"]["scored"] == 0
    assert not (tmp_path / "models").exists()


def test_worker_reports_revision_failure_without_importing_source(tmp_path):
    runner = Path(runtime.__file__).with_name("_native_runner.py")
    request = {
        "schema": 1,
        "source_revision": "wrong",
        "dimension": "Vis",
        "records": [{"id": "sample"}],
    }
    input_file, result_file = tmp_path / "input.json", tmp_path / "result.json"
    input_file.write_text(json.dumps(request), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(runner),
            "--request",
            str(input_file),
            "--result",
            str(result_file),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    document = json.loads(result_file.read_text(encoding="utf-8"))
    assert "revision mismatch" in document["error"]
    assert document["records"][0]["status"] == "failed"
    assert document["records"][0]["raw_score"] is None

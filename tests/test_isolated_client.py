"""Contracts for the optional, dependency-free isolated Ayase client."""

from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
CLIENT_SRC = PROJECT_ROOT / "src"


def _client_module():
    sys.path.insert(0, str(CLIENT_SRC))
    try:
        return importlib.import_module("ayase_client")
    finally:
        sys.path.remove(str(CLIENT_SRC))


def test_regular_ayase_public_api_is_unchanged():
    from ayase import AyasePipeline

    pipeline = AyasePipeline(modules=[])
    assert pipeline.run
    assert pipeline.export
    assert pipeline.results == {}


def test_client_and_protocol_ship_inside_ayase_distribution():
    try:
        import tomllib
    except ImportError:  # Python 3.9/3.10
        import tomli as tomllib

    pyproject = PROJECT_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["name"] == "ayase"
    assert not any(dep.startswith("ayase-protocol") for dep in data["project"]["dependencies"])
    assert data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/ayase",
        "src/ayase_client",
        "src/ayase_protocol",
    ]


def test_client_models_are_protocol_reexports():
    client_models = (CLIENT_SRC / "ayase_client" / "models.py").read_text(encoding="utf-8")
    assert "class Sample" not in client_models
    assert "from ayase_protocol.models import" in client_models


def test_client_and_runtime_versions_stay_in_sync():
    import ayase

    client = _client_module()
    assert client.__version__ == ayase.__version__


def test_remote_sample_preserves_common_sample_interface():
    client = _client_module()
    sample = client.Sample(
        {
            "path": "image.png",
            "is_video": False,
            "reference_path": None,
            "video_metadata": None,
            "image_metadata": {"width": 32, "height": 24},
            "audio_metadata": None,
            "caption": None,
            "quality_metrics": {"blur_score": 12.5},
            "validation_issues": [],
        }
    )

    assert sample.path == Path("image.png")
    assert sample.width == 32
    assert sample.height == 24
    assert sample.is_valid
    assert sample.quality_metrics.blur_score == 12.5
    assert sample.quality_metrics.non_null_metrics() == {"blur_score": 12.5}
    assert sample.quality_metrics.non_null_count() == 1
    assert sample.model_dump(mode="json")["path"] == "image.png"

    copied = client.Sample.model_validate(sample.model_dump()).model_copy(deep=True)
    copied.path = "other.png"
    assert copied.path == Path("other.png")
    assert sample.path == Path("image.png")


def test_runtime_install_is_automatic_and_idempotent(tmp_path, monkeypatch):
    client = _client_module()
    runtime_module = importlib.import_module("ayase_client.runtime")
    monkeypatch.setenv("AYASE_CLIENT_HOME", str(tmp_path))
    monkeypatch.delenv("AYASE_RUNTIME_PYTHON", raising=False)
    calls = []

    def fake_create(_builder, root):
        python = runtime_module._venv_python(Path(root))
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()

    def fake_run(command, check):
        calls.append((command, check))

    monkeypatch.setattr(runtime_module.venv.EnvBuilder, "create", fake_create)
    monkeypatch.setattr(runtime_module.subprocess, "run", fake_run)
    manager = runtime_module.RuntimeManager(client.__version__)

    assert manager.install() == manager.python
    assert manager.install() == manager.python
    assert len(calls) == 1
    assert calls[0][0][-1] == f"ayase=={client.__version__}"


def test_client_to_worker_end_to_end(tmp_path):
    from ayase.isolated_worker import _WorkerServer

    client = _client_module()
    token = "test-token"
    server = _WorkerServer(("127.0.0.1", 0), token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    pipeline = client.AyasePipeline(modules=[], endpoint=endpoint, token=token)
    try:
        source = tmp_path / "sample.png"
        result = pipeline.run(
            tmp_path,
            samples=[{"path": str(source), "is_video": False}],
        )
        assert list(result) == [str(source)]
        assert result[str(source)].path == source
        assert pipeline.results is result
        assert pipeline.stats.total_samples == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

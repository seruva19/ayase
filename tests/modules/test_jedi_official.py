"""Contract tests for the isolated official two-distribution JEDi adapter."""

import json
import subprocess
from pathlib import Path


class _Pipeline:
    def __init__(self):
        self.metrics = {}

    def add_dataset_metric(self, name, value):
        self.metrics[name] = value


def _ready(module):
    module._ml_available = True
    module._backend = "videojedi_isolated"
    module._backend_version = "1.1.0"
    return module


def _assets(root):
    model_dir = root / "models"
    model_dir.mkdir()
    for name in ("vith16.pth.tar", "ssv2-probe.pth.tar"):
        (model_dir / name).write_bytes(name.encode())
    config = root / "config.yaml"
    config.write_text("frames: 16\nresolution: 224\n", encoding="utf-8")
    return model_dir, config


def test_official_jedi_basics():
    from ayase.modules.jedi import OfficialJEDiModule
    from tests.modules.conftest import _test_module_basics

    _test_module_basics(OfficialJEDiModule, "jedi_official")
    assert OfficialJEDiModule.provenance["jedi_score"] == "adapted"
    assert OfficialJEDiModule.provenance["jedi_generated_count"] == "utility"
    assert OfficialJEDiModule.required_packages == []


def test_jedi_generated_documentation_has_correct_score_direction():
    from ayase.metrics_doc import generate_metrics_doc

    document = generate_metrics_doc(run_tests=False)
    section = document.split("### `jedi_score`", 1)[1].split("\n### `", 1)[0]
    assert "↓ lower=better" in section
    assert "↑ higher=better" not in section


def test_routes_two_distributions_to_worker(tmp_path):
    from ayase.models import Sample
    from ayase.modules.jedi import OfficialJEDiModule

    generated = [tmp_path / f"g-{i}.mp4" for i in range(3)]
    references = [tmp_path / f"r-{i}.mp4" for i in range(3)]
    for path in generated + references:
        path.touch()
    model_dir, config = _assets(tmp_path)
    module = _ready(
        OfficialJEDiModule(
            {"num_samples": 2, "model_dir": str(model_dir), "config_path": str(config)}
        )
    )
    calls = []
    module._invoke_worker = lambda payload, check=False: calls.append(payload) or {
        "ok": True,
        "jedi_score": 0.375,
    }
    module.pipeline = _Pipeline()
    for candidate, reference in zip(generated, references):
        module.process(Sample(path=candidate, is_video=True, reference_path=reference))
    module.post_process([])
    payload = calls[0]
    assert payload["reference_paths"] == [str(path.resolve()) for path in references[:2]]
    assert payload["generated_paths"] == [str(path.resolve()) for path in generated[:2]]
    assert payload["num_samples"] == 2
    assert payload["num_frames"] == 16 and payload["resolution"] == 224
    assert module.pipeline.metrics["jedi_score"] == 0.375
    assert len(module.pipeline.metrics["jedi_reference_manifest_sha256"]) == 64


def test_setup_probes_before_download_and_geometry_before_probe(monkeypatch):
    from ayase.modules.jedi import OfficialJEDiModule

    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    monkeypatch.setattr(OfficialJEDiModule, "_global_test_mode", False)
    events = []
    module = OfficialJEDiModule({"backend_python": "X:/jedi/python.exe"})
    module._invoke_worker = lambda payload=None, check=False: events.append(("probe", check)) or (
        _ for _ in ()
    ).throw(RuntimeError("bad env"))
    module._prepare_assets = lambda: events.append(("download", False))
    module.setup()
    assert events == [("probe", True)]
    invalid = OfficialJEDiModule({"num_frames": 8})
    invalid._invoke_worker = lambda *a, **k: (_ for _ in ()).throw(AssertionError("probe ran"))
    invalid._prepare_assets = lambda: (_ for _ in ()).throw(AssertionError("download ran"))
    invalid.setup()
    assert invalid._backend == "unavailable"


def test_invoke_worker_json_environment_error_and_timeout(monkeypatch):
    from ayase.modules.jedi import OfficialJEDiModule
    import ayase.modules.jedi as jedi

    captured = {}

    def success(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        output = Path(command[command.index("--output") + 1])
        request = Path(command[command.index("--input") + 1])
        captured["payload"] = json.loads(request.read_text(encoding="utf-8"))
        output.write_text('{"ok":true,"jedi_score":2.5}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(jedi.subprocess, "run", success)
    module = OfficialJEDiModule(
        {"backend_python": "X:/jedi/python.exe", "backend_cuda_visible_devices": "3", "timeout": 17}
    )
    assert module._invoke_worker({"value": 7})["jedi_score"] == 2.5
    assert captured["payload"] == {"value": 7}
    assert captured["command"][0] == "X:/jedi/python.exe"
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert captured["kwargs"]["timeout"] == 17.0 and captured["kwargs"]["shell"] is False

    def failure(command, **kwargs):
        Path(command[command.index("--output") + 1]).write_text(
            '{"ok":false,"error":"backend failed"}', encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(jedi.subprocess, "run", failure)
    try:
        module._invoke_worker({})
    except RuntimeError as exc:
        assert "backend failed" in str(exc)
    else:
        raise AssertionError("worker failure accepted")
    monkeypatch.setattr(
        jedi.subprocess,
        "run",
        lambda command, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(command, kwargs["timeout"])
        ),
    )
    try:
        module._invoke_worker({})
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError("timeout swallowed")


def test_nonfinite_worker_score_not_published(tmp_path):
    from ayase.models import Sample
    from ayase.modules.jedi import OfficialJEDiModule

    path = tmp_path / "clip.mp4"
    path.touch()
    model_dir, config = _assets(tmp_path)
    module = _ready(OfficialJEDiModule({"model_dir": str(model_dir), "config_path": str(config)}))
    module._invoke_worker = lambda payload, check=False: {"ok": True, "jedi_score": float("nan")}
    module.pipeline = _Pipeline()
    module.process(Sample(path=path, is_video=True, reference_path=path))
    module.post_process([])
    assert module.pipeline.metrics == {} and module.last_metadata == {}


def test_reference_directory_sorted_and_decoder_owned_by_worker(tmp_path, synthetic_video):
    import pytest
    import torch
    from ayase.modules.jedi import OfficialJEDiModule

    reference_dir = tmp_path / "references"
    reference_dir.mkdir()
    for name in ("b.webm", "a.mp4", "notes.txt"):
        (reference_dir / name).touch()
    assert [
        p.name for p in OfficialJEDiModule({"reference_dir": str(reference_dir)})._reference_paths()
    ] == ["a.mp4", "b.webm"]
    pytest.importorskip("av")
    from ayase.modules._jedi_worker import VideoPathDataset

    video, label = VideoPathDataset([synthetic_video], 16, 32)[0]
    assert video.shape == (16, 3, 32, 32) and video.dtype == torch.float32 and label == 0


def test_cache_identity_tracks_collection_weights_and_config(tmp_path):
    from ayase.models import Sample
    from ayase.modules.jedi import OfficialJEDiModule

    generated = tmp_path / "generated.mp4"
    generated.write_bytes(b"video")
    reference = tmp_path / "reference.mp4"
    reference.write_bytes(b"reference")
    model_dir, config = _assets(tmp_path)
    module = _ready(
        OfficialJEDiModule(
            {
                "feature_path": str(tmp_path / "features"),
                "model_dir": str(model_dir),
                "config_path": str(config),
            }
        )
    )
    paths = []
    module._invoke_worker = lambda payload, check=False: paths.append(payload["feature_path"]) or {
        "ok": True,
        "jedi_score": 1.0,
    }
    module.process(Sample(path=generated, is_video=True, reference_path=reference))
    module.post_process([])
    reference.write_bytes(b"changed")
    module.post_process([])
    (model_dir / "vith16.pth.tar").write_bytes(b"replacement")
    module.post_process([])
    config.write_text("change: 1", encoding="utf-8")
    module.post_process([])
    assert len(set(paths)) == 4


def test_test_mode_does_not_probe_or_download():
    from ayase.modules.jedi import OfficialJEDiModule

    module = OfficialJEDiModule({"test_mode": True})
    module._invoke_worker = lambda *a, **k: (_ for _ in ()).throw(AssertionError("probe"))
    module._prepare_assets = lambda: (_ for _ in ()).throw(AssertionError("download"))
    module.setup()
    assert not module._ml_available

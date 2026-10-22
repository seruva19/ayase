"""Tests for the isolated official JEDi worker."""

import json
import sys
from types import SimpleNamespace


def test_video_loader_contract(monkeypatch, tmp_path):
    import numpy as np
    from ayase.modules import _jedi_worker as worker

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")

    class Frame:
        def __init__(self, value):
            self.value = value

        def to_ndarray(self, format):
            assert format == "rgb24"
            return np.full((3, 4, 3), self.value, dtype=np.uint8)

    class Container:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def decode(self, video):
            assert video == 0
            return iter([Frame(0), Frame(127), Frame(255)])

        def seek(self, offset):
            assert offset == 0

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda value: Container()))
    tensor, label = worker.VideoPathDataset([path], 16, 224)[0]
    assert tensor.shape == (16, 3, 224, 224)
    assert tensor.dtype.is_floating_point
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
    assert label == 0


def test_run_payload_routes_official_api(monkeypatch, tmp_path):
    import pytest
    import numpy as np
    from pathlib import Path

    from ayase.modules import _jedi_worker as worker

    generated = tmp_path / "generated.mp4"
    reference = tmp_path / "reference.mp4"
    generated.write_bytes(b"g")
    reference.write_bytes(b"r")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name in ("vith16.pth.tar", "ssv2-probe.pth.tar"):
        (model_dir / name).write_bytes(b"weight")
    config = tmp_path / "config.yaml"
    config.write_text("model: test", encoding="utf-8")
    calls = {}

    class Metric:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self.feature_path = kwargs["feature_path"]

        def load_features(self, **kwargs):
            calls["load"] = kwargs
            for name in ("train.npy", "test.npy"):
                target = Path(self.feature_path) / name
                if not target.exists():
                    np.save(target, np.ones((1, 4), dtype=np.float32))

        def compute_metric(self):
            return 1.25

    class DeviceContext:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return None

    dist_state = {"initialized": False, "destroyed": False}
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            device=lambda value: DeviceContext(),
        ),
        distributed=SimpleNamespace(
            is_initialized=lambda: dist_state["initialized"],
            destroy_process_group=lambda: dist_state.update(
                initialized=False, destroyed=True
            ),
        ),
    )
    monkeypatch.setattr(worker, "package_version", lambda name: "1.1.0")
    monkeypatch.setattr(worker, "make_loader", lambda paths, *args: list(paths))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "videojedi", SimpleNamespace(JEDiMetric=Metric))
    feature_path = tmp_path / "features" / "namespace"
    result = worker.run_payload(
        {
            "generated_paths": [str(generated)],
            "reference_paths": [str(reference)],
            "num_samples": 1,
            "feature_path": str(feature_path),
            "model_dir": str(model_dir),
            "config_path": str(config),
            "num_frames": 16,
            "resolution": 224,
            "device": "cuda:0",
        }
    )
    assert result["jedi_score"] == 1.25
    assert calls["init"]["feature_path"] == str(feature_path.resolve())
    assert calls["load"]["train_loader"] == [reference.resolve()]
    assert calls["load"]["test_loader"] == [generated.resolve()]
    assert calls["load"]["num_samples"] == 1

    Metric.compute_metric = lambda self: float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        worker.run_payload(
            {
                "generated_paths": [str(generated)],
                "reference_paths": [str(reference)],
                "num_samples": 1,
                "feature_path": str(tmp_path / "features-2"),
                "model_dir": str(model_dir),
                "config_path": str(config),
            }
        )

    def failed_load(self, **kwargs):
        dist_state["initialized"] = True
        raise RuntimeError("feature extraction failed")

    Metric.load_features = failed_load
    with pytest.raises(RuntimeError, match="feature extraction failed"):
        worker.run_payload(
            {
                "generated_paths": [str(generated)],
                "reference_paths": [str(reference)],
                "num_samples": 1,
                "feature_path": str(tmp_path / "features-3"),
                "model_dir": str(model_dir),
                "config_path": str(config),
            }
        )
    assert dist_state == {"initialized": False, "destroyed": True}


def test_corrupt_cache_recomputes_only_bad_array_and_retry_reuses_both(tmp_path):
    import numpy as np

    from ayase.modules import _jedi_worker as worker

    cache = tmp_path / "cache"
    cache.mkdir()
    train = cache / "train.npy"
    test = cache / "test.npy"
    np.save(train, np.ones((2, 3), dtype=np.float32))
    test.write_bytes(b"truncated numpy file")
    writes = []

    def mocked_official_load():
        status = worker.repair_feature_cache(cache, 2)
        for name, valid in status.items():
            if not valid:
                np.save(cache / name, np.full((2, 3), 2.0, dtype=np.float32))
                writes.append(name)
        assert all(worker.repair_feature_cache(cache, 2).values())

    mocked_official_load()
    assert writes == ["test.npy"]
    assert np.load(train).tolist() == np.ones((2, 3), dtype=np.float32).tolist()
    mocked_official_load()
    assert writes == ["test.npy"]


def test_cli_writes_success_result_atomically(monkeypatch, tmp_path):
    from ayase.modules import _jedi_worker as worker

    monkeypatch.setattr(worker, "run_payload", lambda payload: {"ok": True, "jedi_score": 1.0})
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text("{}", encoding="utf-8")
    assert worker.main(["--input", str(input_path), "--output", str(output_path)]) == 0
    assert json.loads(output_path.read_text())["ok"] is True


def test_cli_writes_structured_error(tmp_path):
    from ayase.modules import _jedi_worker as worker

    input_path = tmp_path / "bad.json"
    output_path = tmp_path / "result.json"
    input_path.write_text("[]", encoding="utf-8")
    assert worker.main(["--input", str(input_path), "--output", str(output_path)]) == 1
    result = json.loads(output_path.read_text())
    assert result["ok"] is False
    assert "JSON must be an object" in result["error"]


def test_check_rejects_wrong_package_version(monkeypatch):
    from ayase.modules import _jedi_worker as worker

    fake_torch = SimpleNamespace(
        __version__="2.4.1",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            current_device=lambda: 0,
        ),
    )
    monkeypatch.setattr(worker, "package_version", lambda name: "1.0.0")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    result = worker.check_backend()
    assert result["ok"] is False
    assert result["package_supported"] is False
    assert result["required_visible_device"] == "cuda:0"

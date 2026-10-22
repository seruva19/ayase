"""Tests for the persistent isolated module bridge."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ayase.managed_module import (
    ManagedModuleBridge,
    _read_json_after_publish,
    _worker_environment,
    attach_managed_backend,
)
from ayase.models import Sample, ValidationSeverity
from ayase.pipeline import Pipeline


def test_dataset_forwarding_preserves_the_active_module_context(tmp_path):
    from ayase.managed_module import ManagedModuleBridge

    module = SimpleNamespace(name="probe", config={}, process=lambda sample: sample)
    for method in ("process_batch", "post_process", "on_execute", "on_dispose"):
        setattr(module, method, lambda *args: None)
    previous = object()
    observed = []
    pipeline = SimpleNamespace(_active_dataset_module=previous)
    pipeline.add_dataset_metric = lambda key, value: observed.append(
        (key, value, pipeline._active_dataset_module)
    )
    module.pipeline = pipeline
    bridge = ManagedModuleBridge(module, Path(sys.executable), tmp_path)
    bridge._forward_dataset_metrics({"dataset_metrics": {"fvd": 7.5}})
    assert observed == [("fvd", 7.5, module)]
    assert pipeline._active_dataset_module is previous


_BACKEND_SOURCE = """
from ayase.models import QualityMetrics
from ayase.pipeline import PipelineModule

class TinyManagedModule(PipelineModule):
    name = "tiny_managed"

    def setup(self):
        self.ready = True
        self.retained = []

    def on_execute(self):
        self.executing = True

    def process(self, sample):
        if sample.metadata.get("fail"):
            raise RuntimeError("synthetic worker failure")
        self.retained.append(sample)
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.aesthetic_v25_score = 0.5
        sample.metadata["processed"] = self.ready and self.executing
        sample.metadata["marker"] = self.config.get("marker")
        self.pipeline.add_dataset_metric("fvd", 7.5)
        return sample

    def post_process(self, all_samples):
        for sample in self.retained:
            sample.metadata["retained_post_processed"] = True
        for sample in all_samples:
            sample.metadata["post_processed"] = True
        self.pipeline.add_dataset_metric("kvd", 12.5)
"""


def _tiny_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, unavailable: bool = False):
    unique = str(abs(hash(str(tmp_path))))
    module_name = f"tiny_backend_{unique}"
    source = tmp_path / f"{module_name}.py"
    module_source = _BACKEND_SOURCE.replace(
        'name = "tiny_managed"', f'name = "tiny_managed_{unique}"'
    )
    if unavailable:
        module_source = module_source.replace(
            "self.ready = True",
            'self.ready = False\n        self._backend = "unavailable"\n        self._ml_available = False',
        )
    source.write_text(module_source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    current = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(tmp_path), *(item for item in [current] if item)])
    )
    imported = importlib.import_module(module_name)
    return imported.TinyManagedModule(
        {"_ayase_managed_worker": True, "models_dir": str(tmp_path / "models")}
    )


def test_real_worker_preserves_identity_and_forwards_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _tiny_module(tmp_path, monkeypatch)
    pipeline = Pipeline([module])
    module.pipeline = pipeline
    original_process = module.process.__func__
    bridge = ManagedModuleBridge(module, Path(sys.executable), tmp_path / "runtime")
    module._managed_bridge = bridge

    bridge.start()
    module.on_execute()
    sample = Sample(path=tmp_path / "image.png", is_video=False)
    result = module.process(sample)
    assert result is sample
    assert sample.metadata["processed"] is True

    batch = [Sample(path=tmp_path / f"{index}.png", is_video=False) for index in range(2)]
    processed = module.process_batch(batch)
    assert all(left is right for left, right in zip(batch, processed))
    assert pipeline.stats.fvd == 7.5
    module.post_process(batch)
    assert all(item.metadata["post_processed"] for item in batch)
    assert all(item.metadata["retained_post_processed"] for item in batch)
    assert pipeline.stats.kvd == 12.5

    module.on_dispose()
    assert module.process.__func__ is original_process
    assert module._managed_bridge is None
    assert module._mounted is False


def test_process_failure_is_visible_and_returns_original_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _tiny_module(tmp_path, monkeypatch)
    bridge = ManagedModuleBridge(module, Path(sys.executable), tmp_path / "runtime")
    module._managed_bridge = bridge
    bridge.start()
    try:
        sample = Sample(path=tmp_path / "bad.png", is_video=False, metadata={"fail": True})
        assert module.process(sample) is sample
        assert sample.validation_issues[-1].severity == ValidationSeverity.WARNING
        assert "synthetic worker failure" in sample.validation_issues[-1].message
        assert module.name in sample.failed_modules
    finally:
        module.on_dispose()


def test_remote_same_value_write_replays_host_metric_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ayase.models import QualityMetrics, observe_metric_writes

    module = _tiny_module(tmp_path, monkeypatch)
    bridge = ManagedModuleBridge(module, Path(sys.executable), tmp_path / "runtime")
    module._managed_bridge = bridge
    bridge.start()
    try:
        module.on_execute()
        sample = Sample(
            path=tmp_path / "same.png",
            is_video=False,
            quality_metrics=QualityMetrics(aesthetic_v25_score=0.5),
        )
        written = set()
        with observe_metric_writes(lambda metrics, field: written.add(field)):
            module.process(sample)
        assert "aesthetic_v25_score" in written
    finally:
        module.on_dispose()


def test_worker_rejects_known_unavailable_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _tiny_module(tmp_path, monkeypatch, unavailable=True)
    bridge = ManagedModuleBridge(module, Path(sys.executable), tmp_path / "runtime")
    module._managed_bridge = bridge

    with pytest.raises(RuntimeError, match="backend='unavailable'.*ml_available=False"):
        bridge.start()
    bridge.dispose()


def test_workers_sharing_interpreter_keep_config_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _tiny_module(tmp_path, monkeypatch)
    second = type(first)({"_ayase_managed_worker": True, "marker": "second"})
    first.config["marker"] = "first"
    first_bridge = ManagedModuleBridge(first, Path(sys.executable), tmp_path / "worker-first")
    second_bridge = ManagedModuleBridge(second, Path(sys.executable), tmp_path / "worker-second")
    first._managed_bridge = first_bridge
    second._managed_bridge = second_bridge
    first_bridge.start()
    second_bridge.start()
    try:
        first.on_execute()
        second.on_execute()
        first_sample = Sample(path=tmp_path / "first.png", is_video=False)
        second_sample = Sample(path=tmp_path / "second.png", is_video=False)
        first.process(first_sample)
        second.process(second_sample)
        assert first_sample.metadata["marker"] == "first"
        assert second_sample.metadata["marker"] == "second"
    finally:
        first.on_dispose()
        second.on_dispose()


def test_attach_skips_test_mode_without_provisioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _tiny_module(tmp_path, monkeypatch)
    module.config["test_mode"] = True
    monkeypatch.setattr(
        "ayase.managed_module.ensure_managed_environment",
        lambda *args, **kwargs: pytest.fail("test mode must not provision"),
    )
    assert attach_managed_backend(module) is False


def test_protocol_read_retries_one_transient_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "reply.json"
    path.write_text('{"ok": true}', encoding="utf-8")
    original = Path.read_text
    attempts = 0

    def flaky_read_text(self, *args, **kwargs):
        nonlocal attempts
        if self == path and attempts == 0:
            attempts += 1
            raise PermissionError("synthetic Windows scanner lock")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)
    assert _read_json_after_publish(path, 10**20) == {"ok": True}
    assert attempts == 1


def test_worker_config_rejects_unsupported_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _tiny_module(tmp_path, monkeypatch)
    module.config["callback"] = lambda: None
    bridge = ManagedModuleBridge(module, Path(sys.executable), tmp_path / "runtime")
    with pytest.raises(TypeError, match="config.callback contains unsupported function"):
        bridge.start()


def test_attach_uses_dependency_spec_and_sets_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _tiny_module(tmp_path, monkeypatch)
    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    module._global_test_mode = False
    module.config.pop("_ayase_managed_worker")
    module.config["models_dir"] = str(tmp_path / "models")
    dependency = SimpleNamespace(
        requirements=("backend-package==1.0",),
        python_version="3.10.15",
        pip_options=("--no-index",),
        source_roots=(),
    )
    observed = {}

    def fake_ensure(name, requirements, models_dir, **kwargs):
        observed.update(name=name, requirements=requirements, models_dir=models_dir, kwargs=kwargs)
        return Path(sys.executable)

    class FakeBridge:
        def __init__(self, target, interpreter, runtime_root):
            self.target = target

        def start(self):
            return None

        def dispose(self):
            return None

    monkeypatch.setattr("ayase.managed_module._dependency_for", lambda name: dependency)
    monkeypatch.setattr("ayase.managed_module.ensure_managed_environment", fake_ensure)
    monkeypatch.setattr("ayase.managed_module.ManagedModuleBridge", FakeBridge)

    assert attach_managed_backend(module) is True
    assert observed["kwargs"]["python_version"] == "3.10.15"
    assert observed["kwargs"]["pip_options"][:2] == (
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu124",
    )
    assert any(item.startswith("pydantic") for item in observed["requirements"])
    assert "backend-package==1.0" in observed["requirements"]
    assert module._backend == "managed:unknown"
    assert module._mounted is True


def test_platform_unsupported_dependency_falls_back_without_provisioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _tiny_module(tmp_path, monkeypatch)
    module.config.pop("_ayase_managed_worker")
    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    module._global_test_mode = False
    dependency = SimpleNamespace(
        config_conditions=(),
        unsupported_reason=None,
        unsupported_platforms=((sys.platform, "not available here"),),
    )
    monkeypatch.setattr("ayase.managed_module._dependency_for", lambda name: dependency)
    monkeypatch.setattr(
        "ayase.managed_module.ensure_managed_environment",
        lambda *args, **kwargs: pytest.fail("unsupported platform must use host fallback"),
    )
    assert attach_managed_backend(module) is False


@pytest.mark.skipif(os.name != "nt", reason="Windows conda DLL layout")
def test_native_worker_environment_prepends_conda_dll_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _tiny_module(tmp_path, monkeypatch)
    prefix = tmp_path / "conda"
    (prefix / "conda-meta").mkdir(parents=True)
    interpreter = prefix / "python.exe"

    environment = _worker_environment(module, interpreter)

    assert environment["CONDA_PREFIX"] == str(prefix)
    path_entries = environment["PATH"].split(os.pathsep)
    assert path_entries[:3] == [
        str(prefix / "Library" / "bin"),
        str(prefix / "Scripts"),
        str(prefix),
    ]

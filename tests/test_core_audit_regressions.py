"""Regression checks for disposal and relative filter symlink targets."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ayase.models import Sample
from ayase.pipeline import Pipeline, PipelineModule


class _LifecycleProbe(PipelineModule):
    name = "core_audit_lifecycle_probe"
    provenance = "utility"

    def __init__(self, config=None):
        super().__init__(config)
        self.setups = 0
        self.resource = None

    def setup(self):
        self.setups += 1
        self.resource = object()

    def teardown(self):
        self.resource = None

    def process(self, sample):
        assert self.resource is not None
        return sample


def test_pipeline_reloads_disposed_resources(monkeypatch):
    monkeypatch.setattr(PipelineModule, "_global_test_mode", False)
    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    module = _LifecycleProbe()
    pipeline = Pipeline([module])
    for run in range(2):
        pipeline.start()
        sample = pipeline.process_sample(Sample(path=Path(f"{run}.png"), is_video=False))
        assert sample.failed_modules == []
        pipeline.stop()
        assert not module._mounted
    assert module.setups == 2


def test_pipeline_resets_mount_even_if_disposal_override_raises(monkeypatch):
    module = _LifecycleProbe()
    pipeline = Pipeline([module])
    pipeline.start()

    def fail_disposal():
        raise RuntimeError("disposal failed")

    monkeypatch.setattr(module, "on_dispose", fail_disposal)
    pipeline.stop()
    assert not module._mounted
    assert "disposal failed" in pipeline.module_failures[module.name]


def test_direct_disposal_resets_mount_when_teardown_raises(monkeypatch):
    module = _LifecycleProbe()
    module._mounted = True

    def fail_teardown():
        raise RuntimeError("teardown failed")

    monkeypatch.setattr(module, "teardown", fail_teardown)
    with pytest.raises(RuntimeError, match="teardown failed"):
        module.on_dispose()
    assert not module._mounted


def test_filter_relative_dataset_symlink_targets_source(tmp_path, monkeypatch):
    import ayase.cli as cli

    monkeypatch.chdir(tmp_path)
    source = Path("dataset/nested/clip.png")
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image")
    sample = Sample(path=source, is_video=False)
    calls = []

    monkeypatch.setattr(cli, "scan_dataset", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(cli, "_select_modules", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli, "_instantiate_modules", lambda *args, **kwargs: [])

    def run(pipeline, samples):
        pipeline.results = {str(source): sample}

    def symlink(target, destination):
        calls.append((Path(target), Path(destination)))

    monkeypatch.setattr(cli, "_run_pipeline", run)
    monkeypatch.setattr(cli.os, "symlink", symlink)
    result = CliRunner().invoke(
        cli.app, ["filter", "dataset", "--mode", "symlink", "--output", "output"]
    )
    assert result.exit_code == 0, result.output
    assert calls == [(source.resolve(), Path("output/nested/clip.png"))]
    assert calls[0][0].read_bytes() == b"image"

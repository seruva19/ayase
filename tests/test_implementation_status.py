from rich.console import Console

from ayase.metric_catalog import build_metric_catalog, render_metric_help
from ayase.metrics_doc import compute_external_backend_partition
from ayase.models import Sample
from ayase.pipeline import ModuleRegistry, Pipeline, PipelineModule


class _ImplementedModule(PipelineModule):
    name = "_status_implemented"
    description = "Implemented test module"
    implementation_notes = "Uses the production path."

    def process(self, sample):
        return sample


class _ApproximateModule(PipelineModule):
    name = "_status_approximate"
    description = "Approximate test module"
    implementation_status = "approximate"
    implementation_notes = "Uses a documented approximation."

    def process(self, sample):
        return sample


class _ExplicitUnimplementedModule(PipelineModule):
    name = "_status_unimplemented"
    description = "Unimplemented test module"
    implementation_status = "unimplemented"

    def process(self, sample):
        return sample


class _LegacyExternalModule(PipelineModule):
    name = "_status_legacy_external"
    description = "Legacy external test module"
    requires_external_backend = True

    def process(self, sample):
        return sample


class _InvalidStatusModule(PipelineModule):
    description = "Invalid status test module"
    implementation_status = "partial"

    def process(self, sample):
        return sample


def test_status_resolution_preserves_legacy_external_behavior():
    assert _ImplementedModule.resolved_implementation_status() == "implemented"
    assert _ApproximateModule.resolved_implementation_status() == "approximate"
    assert _ExplicitUnimplementedModule.resolved_implementation_status() == "unimplemented"
    assert _LegacyExternalModule.resolved_implementation_status() == "unimplemented"


def test_invalid_status_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="Invalid implementation_status"):
        _InvalidStatusModule.resolved_implementation_status()


def test_metadata_contains_resolved_status_notes_and_cache_tracks_them():
    first = _ApproximateModule.get_metadata()
    assert first["implementation_status"] == "approximate"
    assert first["implementation_notes"] == "Uses a documented approximation."

    original = _ApproximateModule.implementation_notes
    try:
        _ApproximateModule.implementation_notes = "Updated readiness note."
        assert _ApproximateModule.get_metadata()["implementation_notes"] == (
            "Updated readiness note."
        )
    finally:
        _ApproximateModule.implementation_notes = original


def test_fast_and_detailed_catalog_agree_and_help_renders_readiness():
    names = [_ApproximateModule.name, _LegacyExternalModule.name]
    fast = build_metric_catalog(names, detailed=False)
    detailed = build_metric_catalog(names, detailed=True)
    assert [(m.name, m.implementation_status) for m in fast.modules] == [
        (m.name, m.implementation_status) for m in detailed.modules
    ]

    console = Console(record=True, width=120)
    assert render_metric_help(console, detailed, _ApproximateModule.name)
    output = console.export_text()
    assert "Readiness" in output
    assert "approximate" in output
    assert "Uses a documented approximation." in output


def test_unimplemented_is_excluded_but_approximate_is_delivered():
    names = {
        _ImplementedModule.name,
        _ApproximateModule.name,
        _ExplicitUnimplementedModule.name,
        _LegacyExternalModule.name,
    }
    unimplemented, _ = compute_external_backend_partition(names)
    assert unimplemented == {
        _ExplicitUnimplementedModule.name,
        _LegacyExternalModule.name,
    }

    delivered = ModuleRegistry.list_modules(include_external_backends=False)
    assert _ImplementedModule.name in delivered
    assert _ApproximateModule.name in delivered
    assert _ExplicitUnimplementedModule.name not in delivered
    assert _LegacyExternalModule.name not in delivered


def test_modules_list_shows_readiness_separately_from_provenance(monkeypatch):
    from typer.testing import CliRunner
    from ayase import cli

    classes = [_ImplementedModule, _ApproximateModule, _ExplicitUnimplementedModule]
    monkeypatch.setattr(cli, "_discover_all_modules", lambda config: None)
    monkeypatch.setattr(
        ModuleRegistry, "list_modules", lambda **kwargs: {cls.name: cls.description for cls in classes}
    )
    result = CliRunner().invoke(cli.app, ["modules", "list"], env={"COLUMNS": "240"})
    assert result.exit_code == 0, result.output
    assert "Readiness" in result.output
    assert "Provenance" in result.output
    assert "approximate" in result.output
    assert "unimplemented" in result.output


def test_explicit_unimplemented_never_mounts_or_processes(tmp_path):
    class NeverMountModule(PipelineModule):
        name = "_status_never_mount"
        description = "Must remain inert"
        implementation_status = "unimplemented"

        def setup(self):
            raise AssertionError("setup must not be called")

        def process(self, sample):
            raise AssertionError("process must not be called")

    module = NeverMountModule()
    pipeline = Pipeline([module])
    pipeline.start()
    sample = Sample(path=tmp_path / "sample.png", is_video=False)
    assert pipeline.process_sample(sample) is sample
    assert sample.quality_metrics is None
    assert module._mounted is False
    assert pipeline.get_run_status()["availability_excluded"] == {
        module.name: "implementation_unavailable"
    }

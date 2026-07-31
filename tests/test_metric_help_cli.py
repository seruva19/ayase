"""CLI contracts for interactive metric and module help."""

import re

from typer.testing import CliRunner

from ayase.cli import app
from ayase.metric_catalog import build_metric_catalog
from ayase.pipeline import ModuleRegistry


runner = CliRunner()


def test_help_lists_available_metrics() -> None:
    result = runner.invoke(app, ["help"])

    assert result.exit_code == 0, result.output
    assert "Available Ayase Metrics" in result.output
    assert "rqvqa_score" in result.output
    assert "rqvqa" in result.output
    assert "ayase help <metric-or-module>" in result.output


def test_help_module_shows_models_config_and_usage() -> None:
    # A wide console on purpose. Long asset ids are folded across rows when the
    # table does not fit, and a folded value is interleaved with the other
    # columns, so no amount of post-processing can put it back together. Without
    # a fixed width this assertion silently tested the terminal size instead of
    # the help output.
    result = runner.invoke(app, ["help", "rqvqa"], env={"COLUMNS": "240"})

    assert result.exit_code == 0, result.output
    assert "RQ-VQA rich quality-aware blind VQA ensemble" in result.output
    assert "rqvqa_score" in result.output
    compact_output = re.sub(r"\s+", "", result.output)
    assert "SLOWFAST_8x8_R50.pyth" in compact_output
    assert "q-future/one-align" in result.output
    assert "ensemble_size" in result.output
    assert "ayase scan MEDIA_PATH --modules rqvqa" in result.output

def test_help_metric_resolves_owning_module() -> None:
    result = runner.invoke(app, ["help", "rqvqa_score"])

    assert result.exit_code == 0, result.output
    assert "Metric rqvqa_score is produced by 1 module" in result.output
    assert "rqvqa" in result.output
    assert "higher=better" in result.output


def test_help_unknown_name_returns_suggestions() -> None:
    result = runner.invoke(app, ["help", "rqvqa_scor"])

    assert result.exit_code == 2
    assert "Unknown metric or module" in result.output
    assert "rqvqa_score" in result.output


def test_catalog_does_not_instantiate_modules(monkeypatch) -> None:
    from ayase.modules.rqvqa import RQVQAModule

    ModuleRegistry.discover_modules()

    def fail_init(*args, **kwargs):
        raise AssertionError("help must not instantiate or mount modules")

    monkeypatch.setattr(RQVQAModule, "__init__", fail_init)
    catalog = build_metric_catalog(["rqvqa"])

    module = catalog.module("rq-vqa")
    assert module is not None
    assert any(metric.name == "rqvqa_score" for metric in module.metrics)
    assert any(model.name == "q-future/one-align" for model in module.models)

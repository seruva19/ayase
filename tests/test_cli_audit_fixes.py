"""Regression tests for CLI validation that must happen before pipeline execution."""

from pathlib import Path

import pytest
import typer
from PIL import Image
from typer.testing import CliRunner

from ayase.cli import _instantiate_modules, _parse_pipeline_str, app
from ayase.config import AyaseConfig
from ayase.pipeline import ModuleRegistry


def test_parse_pipeline_rejects_empty_explicit_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ayase.cli._discover_all_modules", lambda config: None)

    with pytest.raises(typer.Exit) as exc_info:
        _parse_pipeline_str(" , ", AyaseConfig())

    assert exc_info.value.exit_code == 1


def test_parse_pipeline_rejects_unknown_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ayase.cli._discover_all_modules", lambda config: None)
    monkeypatch.setattr(ModuleRegistry, "get_module", lambda name: None)

    with pytest.raises(typer.Exit) as exc_info:
        _parse_pipeline_str("does_not_exist", AyaseConfig())

    assert exc_info.value.exit_code == 1


def test_parse_pipeline_rejects_constructor_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenModule:
        def __init__(self, config: object) -> None:
            raise RuntimeError("constructor failed")

    monkeypatch.setattr("ayase.cli._discover_all_modules", lambda config: None)
    monkeypatch.setattr(ModuleRegistry, "get_module", lambda name: BrokenModule)

    with pytest.raises(typer.Exit) as exc_info:
        _parse_pipeline_str("broken", AyaseConfig())

    assert exc_info.value.exit_code == 1


def test_instantiate_modules_rejects_unknown_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ModuleRegistry, "get_module", lambda name: None)

    with pytest.raises(typer.Exit) as exc_info:
        _instantiate_modules(["does_not_exist"], AyaseConfig())

    assert exc_info.value.exit_code == 1


def test_instantiate_modules_rejects_constructor_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenModule:
        def __init__(self, config: object) -> None:
            raise RuntimeError("constructor failed")

    monkeypatch.setattr(ModuleRegistry, "get_module", lambda name: BrokenModule)

    with pytest.raises(typer.Exit) as exc_info:
        _instantiate_modules(["broken"], AyaseConfig())

    assert exc_info.value.exit_code == 1


@pytest.mark.parametrize(
    ("command", "expected_message"),
    [
        (["run", "clip.jpg", "--pipeline", "metadata", "--format", "xml"], "Unknown format: xml"),
        (["scan", "dataset", "--format", "xml"], "Unknown format: xml"),
        (["stats", "dataset", "--format", "xml"], "Unknown format: xml"),
    ],
)
def test_invalid_format_fails_before_pipeline_execution(
    monkeypatch: pytest.MonkeyPatch, command: list[str], expected_message: str
) -> None:
    monkeypatch.setattr(
        "ayase.cli._run_pipeline",
        lambda *args, **kwargs: pytest.fail("pipeline must not execute"),
    )

    result = CliRunner().invoke(app, command)

    assert result.exit_code == 1
    assert expected_message in result.output


def test_unknown_cli_module_fails_before_pipeline_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (2, 2)).save(dataset / "sample.png")
    monkeypatch.setattr("ayase.cli._discover_all_modules", lambda config: None)
    monkeypatch.setattr(ModuleRegistry, "get_module", lambda name: None)
    monkeypatch.setattr(
        "ayase.cli._run_pipeline",
        lambda *args, **kwargs: pytest.fail("pipeline must not execute"),
    )

    result = CliRunner().invoke(
        app, ["scan", str(dataset), "--modules", "does_not_exist"]
    )

    assert result.exit_code == 1
    assert "Unknown module: does_not_exist" in result.output


def test_cli_constructor_error_fails_before_pipeline_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2)).save(image_path)

    class BrokenModule:
        def __init__(self, config: object) -> None:
            raise RuntimeError("constructor failed")

    monkeypatch.setattr("ayase.cli._discover_all_modules", lambda config: None)
    monkeypatch.setattr(ModuleRegistry, "get_module", lambda name: BrokenModule)
    monkeypatch.setattr(
        "ayase.cli._run_pipeline",
        lambda *args, **kwargs: pytest.fail("pipeline must not execute"),
    )

    result = CliRunner().invoke(
        app, ["run", str(image_path), "--pipeline", "broken"]
    )

    assert result.exit_code == 1
    assert "Error initializing module 'broken': constructor failed" in result.output


def test_run_creates_output_parent_before_pipeline_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "nested" / "report.json"

    def assert_parent_exists(*args: object, **kwargs: object) -> int:
        assert output.parent.is_dir()
        return 0

    monkeypatch.setattr("ayase.cli._parse_pipeline_str", lambda *args, **kwargs: [])
    monkeypatch.setattr("ayase.cli._run_pipeline", assert_parent_exists)

    result = CliRunner().invoke(
        app,
        ["run", "missing.jpg", "--pipeline", "metadata", "--output", str(output)],
    )

    assert result.exit_code == 0
    assert not output.exists()


def test_tui_exception_returns_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    from ayase.tui import AyaseApp

    monkeypatch.setattr(AyaseApp, "run", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    result = CliRunner().invoke(app, ["tui"])

    assert result.exit_code == 1
    assert "Unexpected error: boom" in result.output

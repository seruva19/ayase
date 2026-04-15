"""CLI interface for Ayase using Typer."""


import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import re
from typing import Optional, Iterable, List, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from . import __version__
from .config import AyaseConfig
from .pipeline import Pipeline, ModuleRegistry, PipelineModule
from .scanner import DatasetScanner, scan_dataset, sample_from_path
from .models import Sample

app = typer.Typer(
    name="ayase",
    help="Modular media quality metrics toolkit",
    no_args_is_help=True,
)
console = Console()


def _discover_all_modules(config: AyaseConfig) -> None:
    plugin_paths = []
    if config and getattr(config, "pipeline", None):
        plugin_paths = config.pipeline.plugin_folders
    ModuleRegistry.discover_modules(plugin_paths=plugin_paths)


def _print_readiness() -> None:
    readiness = ModuleRegistry.readiness_report()
    if not readiness:
        return
    table = Table(title="Module Readiness")
    table.add_column("Module")
    table.add_column("Status")
    table.add_column("Error")
    for name, info in sorted(readiness.items()):
        status = "READY" if info.get("status") == "ready" else "MISSING"
        error = info.get("error") or ""
        table.add_row(name, status, error)
    console.print(table)


def _select_modules(quick: bool, deep: bool, config: AyaseConfig) -> List[str]:
    _discover_all_modules(config)
    all_modules = list(ModuleRegistry.list_modules().keys())
    if deep:
        return sorted(all_modules)
    if quick:
        return [name for name in ["metadata", "basic_quality"] if name in all_modules]
    if config.pipeline.modules:
        return [name for name in config.pipeline.modules if name in all_modules]
    # Default "balanced" set
    preferred = [
        "metadata",
        "basic_quality",
        "structural",
        "motion",
        "text_detection",
        "watermark_classifier",
        "aesthetic_scoring",
        "video_text_matching",
    ]
    return [name for name in preferred if name in all_modules]


def _parse_pipeline_str(pipeline_str: str, config: AyaseConfig) -> List[PipelineModule]:
    """Parse a pipeline string like 'metadata,motion{sample_rate=10}'."""
    _discover_all_modules(config)
    modules = []

    # Simple regex to split by comma but ignore commas inside curly braces
    parts = re.split(r",(?![^{]*})", pipeline_str)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        params = {}
        if "{" in part and part.endswith("}"):
            name, param_str = part[:-1].split("{", 1)
            name = name.strip()
            # Parse params: key=value
            for pair in param_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    # Try to convert to int/float/bool
                    if v.lower() == "true":
                        v = True
                    elif v.lower() == "false":
                        v = False
                    else:
                        try:
                            if "." in v:
                                v = float(v)
                            else:
                                v = int(v)
                        except ValueError:
                            pass
                    params[k] = v
        else:
            name = part

        module_cls = ModuleRegistry.get_module(name)
        if module_cls:
            try:
                if "models_dir" not in params and config:
                    params["models_dir"] = str(config.general.models_dir)
                if "parallel_jobs" not in params and config:
                    params["parallel_jobs"] = config.general.parallel_jobs
                modules.append(module_cls(config=params))
            except Exception as e:
                console.print(f"[red]Error initializing module '{name}': {e}[/red]")
        else:
            console.print(f"[yellow]Warning: Module '{name}' not found.[/yellow]")

    return modules


def _process_samples(
    pipeline: Pipeline, samples: Iterable[Sample],
) -> int:
    processed_count = 0
    for sample in samples:
        pipeline.process_sample(sample)
        processed_count += 1
    return processed_count


def _run_pipeline(pipeline: Pipeline, samples: Iterable[Sample]) -> int:
    """Run a pipeline over samples and always dispose mounted modules."""
    started = False
    try:
        pipeline.start()
        started = True
        return _process_samples(pipeline, samples)
    finally:
        if started:
            pipeline.stop()


def _iter_dataset_samples(
    dataset_path: Path,
    *,
    include_videos: bool = True,
    include_images: bool = True,
    recursive: bool = True,
) -> Iterable[Sample]:
    scanner = DatasetScanner(
        dataset_path=dataset_path,
        include_videos=include_videos,
        include_images=include_images,
        recursive=recursive,
    )
    return scanner.scan()


def _iter_input_samples(paths: Iterable[Path], *, recursive: bool) -> Iterable[Sample]:
    for path in paths:
        if path.is_dir():
            yield from _iter_dataset_samples(
                path,
                include_videos=True,
                include_images=True,
                recursive=recursive,
            )
            continue

        sample = sample_from_path(path)
        if sample is None:
            console.print(f"[red]Unsupported file type: {path}[/red]")
            continue
        yield sample


def _write_markdown_report(pipeline: Pipeline) -> str:
    stats = pipeline.stats
    lines = [
        "# Ayase Report",
        "",
        f"- Total samples: {stats.total_samples}",
        f"- Valid samples: {stats.valid_samples}",
        f"- Invalid samples: {stats.invalid_samples}",
        "",
        "## Issues",
        "",
    ]
    for sample in pipeline.results.values():
        if not sample.validation_issues:
            continue
        lines.append(f"### {sample.path}")
        for issue in sample.validation_issues:
            lines.append(f"- {issue.severity.value.upper()}: {issue.message}")
        lines.append("")
    return "\n".join(lines).strip()


def _export_artifacts(pipeline: Pipeline, config: AyaseConfig, label: str) -> Optional[Path]:
    output_dir = Path(config.output.artifacts_dir) if config and config.output else Path("reports")
    fmt = (config.output.artifacts_format if config and config.output else None) or "json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"ayase_{label}_{timestamp}.{fmt}"
        pipeline.export_report(output_path, format=fmt)
        console.print(f"[green]Artifact saved:[/green] {output_path}")
        return output_path
    except Exception as e:
        console.print(f"[yellow]Artifact export failed: {e}[/yellow]")
        return None


def _instantiate_modules(module_names: List[str], config: AyaseConfig) -> List[PipelineModule]:
    modules = []
    for name in module_names:
        module_cls = ModuleRegistry.get_module(name)
        if not module_cls:
            continue
        params = {
            "models_dir": str(config.general.models_dir),
            "parallel_jobs": config.general.parallel_jobs,
        }
        try:
            modules.append(module_cls(config=params))
        except Exception as e:
            console.print(f"[red]Error initializing module '{name}': {e}[/red]")
    return modules


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold]Ayase[/bold] version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-V", callback=version_callback, is_eager=True),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Ayase - Modular media quality metrics toolkit."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
        console.print("[dim]Verbose mode enabled[/dim]")


@app.command()
def scan(
    dataset_path: Annotated[
        Optional[Path], typer.Argument(help="Path to dataset directory")
    ] = None,
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Output report file (default: stdout)")
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Report format: json|csv|markdown|html"),
    ] = "markdown",
    pipeline: Annotated[
        Optional[str],
        typer.Option(
            "--pipeline",
            "-p",
            help="Explicit pipeline modules (e.g. 'metadata,motion{sample_rate=10}')",
        ),
    ] = None,
    modules_flag: Annotated[
        Optional[str],
        typer.Option(
            "--modules",
            help="Comma-separated module names (e.g. 'metadata,basic_quality,motion')",
        ),
    ] = None,
    jobs: Annotated[
        Optional[int],
        typer.Option("--jobs", "-j", help="Parallel job hint for capable modules/backends"),
    ] = None,
    quick: Annotated[
        bool,
        typer.Option("--quick", help="Quick scan (skip quality metrics)"),
    ] = False,
    deep: Annotated[
        bool,
        typer.Option("--deep", help="Deep scan (all quality metrics, slower)"),
    ] = False,
) -> None:
    """Scan a dataset and generate a quality metrics report."""
    is_quiet = format == "json" and output is None

    if not is_quiet:
        console.print(f"[bold blue]Scanning dataset:[/bold blue] {dataset_path}")
        console.print(f"[bold]Output format:[/bold] {format}")

        if jobs:
            console.print(f"[bold]Parallel jobs:[/bold] {jobs}")
        if quick:
            console.print("[yellow]Quick scan mode enabled[/yellow]")
        if deep:
            console.print("[cyan]Deep scan mode enabled[/cyan]")

    config = AyaseConfig.load()
    if jobs is not None:
        config.general.parallel_jobs = jobs
    if dataset_path is None:
        dataset_path = config.pipeline.dataset_path
    if dataset_path is None:
        console.print("[red]Dataset path is required.[/red]")
        raise typer.Exit(code=1)

    if pipeline:
        modules = _parse_pipeline_str(pipeline, config)
    elif modules_flag:
        _discover_all_modules(config)
        module_names = [n.strip() for n in modules_flag.split(",") if n.strip()]
        modules = _instantiate_modules(module_names, config)
    else:
        module_names = _select_modules(quick, deep, config)
        modules = _instantiate_modules(module_names, config)

    p = Pipeline(modules)
    samples = _iter_dataset_samples(dataset_path, include_videos=True, include_images=True)
    _run_pipeline(p, samples)

    if format == "markdown":
        report = _write_markdown_report(p)
        if output:
            output.write_text(report, encoding="utf-8")
        else:
            console.print(report)
        return

    if output:
        p.export_report(output, format=format)
    else:
        if format == "json":
            data = {
                "stats": p.stats.model_dump(),
                "samples": [s.model_dump(mode="json") for s in p.results.values()],
            }
            console.print_json(data=data)
        elif format == "csv":
            writer = csv.writer(sys.stdout)
            writer.writerow(["Path", "Valid", "Issues", "Recommendations", "Technical Score"])
            for s in p.results.values():
                issues_str = "; ".join([i.message for i in s.validation_issues])
                recs_str = "; ".join(
                    [i.recommendation for i in s.validation_issues if i.recommendation]
                )
                score = (s.quality_metrics.technical_score if s.quality_metrics else None) or 0.0
                writer.writerow([str(s.path), s.is_valid, issues_str, recs_str, f"{score:.2f}"])
        elif format == "html":
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as tmp:
                tmp_path = Path(tmp.name)
            try:
                p.export_report(tmp_path, format="html")
                console.print(tmp_path.read_text(encoding="utf-8"))
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            console.print(f"[red]Unknown format: {format}[/red]")
            raise typer.Exit(code=1)


@app.command()
def run(
    paths: Annotated[List[Path], typer.Argument(help="Paths to files or directories")],
    pipeline: Annotated[
        str,
        typer.Option(
            "--pipeline",
            "-p",
            help="Pipeline configuration (e.g. 'metadata,motion{sample_rate=10}')",
        ),
    ],
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json|csv|markdown"),
    ] = "json",
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file")] = None,
    recursive: Annotated[
        bool,
        typer.Option("--recursive/--no-recursive", help="Scan directories recursively"),
    ] = True,
) -> None:
    """Run a specific quality assessment pipeline on target paths."""
    config = AyaseConfig.load()
    modules = _parse_pipeline_str(pipeline, config)
    p = Pipeline(modules)

    processed_count = _run_pipeline(p, _iter_input_samples(paths, recursive=recursive))
    if processed_count == 0:
        console.print("[yellow]No valid files found to process.[/yellow]")
        raise typer.Exit(code=0)

    if format == "json":
        data = {
            "stats": p.stats.model_dump(),
            "samples": [s.model_dump(mode="json") for s in p.results.values()],
        }
        if output:
            output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            console.print_json(data=data)
    elif format == "markdown":
        report = _write_markdown_report(p)
        if output:
            output.write_text(report, encoding="utf-8")
        else:
            console.print(report)
    elif format == "csv":
        if output:
            p.export_report(output, format="csv")
        else:
            writer = csv.writer(sys.stdout)
            writer.writerow(["Path", "Valid", "Issues", "Technical Score"])
            for s in p.results.values():
                issues = "; ".join([i.message for i in s.validation_issues])
                score = (s.quality_metrics.technical_score if s.quality_metrics else None) or 0.0
                writer.writerow([str(s.path), s.is_valid, issues, f"{score:.2f}"])
    else:
        console.print(f"[red]Unknown format: {format}[/red]")
        raise typer.Exit(code=1)


@app.command()
def filter(
    dataset_path: Annotated[Path, typer.Argument(help="Path to dataset directory")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output directory")] = None,
    min_score: Annotated[
        Optional[int],
        typer.Option("--min-score", help="Minimum metric score [0-100]"),
    ] = None,
    metric: Annotated[
        str,
        typer.Option("--metric", help="Metric to filter by (default: technical_score)"),
    ] = "technical_score",
    mode: Annotated[
        str,
        typer.Option("--mode", help="Filter mode: symlink|copy|list"),
    ] = "list",
    aspect_ratio: Annotated[
        Optional[str],
        typer.Option("--aspect-ratio", help="Filter by aspect ratio (e.g., 16:9)"),
    ] = None,
    resolution: Annotated[
        Optional[str],
        typer.Option("--resolution", help="Filter by resolution (e.g., 1280x720)"),
    ] = None,
) -> None:
    """Filter dataset based on quality metrics."""
    console.print(f"[bold magenta]Filtering dataset:[/bold magenta] {dataset_path}")
    if output is not None:
        console.print(f"[bold]Output:[/bold] {output}")
    console.print(f"[bold]Mode:[/bold] {mode}")

    if min_score is not None:
        console.print(f"[bold]Minimum {metric}:[/bold] {min_score}")
    if aspect_ratio:
        console.print(f"[bold]Aspect ratio:[/bold] {aspect_ratio}")
    if resolution:
        console.print(f"[bold]Resolution:[/bold] {resolution}")

    config = AyaseConfig.load()
    module_names = _select_modules(quick=False, deep=False, config=config)
    modules = _instantiate_modules(module_names, config)
    pipeline = Pipeline(modules)

    samples = _iter_dataset_samples(dataset_path, include_videos=True, include_images=True)
    _run_pipeline(pipeline, samples)

    target_ar = None
    if aspect_ratio:
        try:
            w, h = aspect_ratio.split(":")
            target_ar = float(w) / float(h)
        except Exception:
            console.print("[red]Invalid aspect ratio format; expected W:H[/red]")
            raise typer.Exit(code=1)

    target_res = None
    if resolution:
        try:
            w, h = resolution.lower().split("x")
            target_res = (int(w), int(h))
        except Exception:
            console.print("[red]Invalid resolution format; expected WxH[/red]")
            raise typer.Exit(code=1)

    candidates = []
    for sample in pipeline.results.values():
        if min_score is not None:
            score = 0.0
            if sample.quality_metrics:
                score = getattr(sample.quality_metrics, metric, None) or 0.0
            if score < min_score:
                continue
        if target_ar and sample.aspect_ratio and abs(sample.aspect_ratio - target_ar) > 0.01:
            continue
        if target_res and (sample.width is None or sample.height is None or (sample.width, sample.height) != target_res):
            continue
        candidates.append(sample)

    if mode == "list":
        for sample in candidates:
            console.print(str(sample.path))
        return

    if not output:
        console.print("[red]Output directory required for copy/symlink modes.[/red]")
        raise typer.Exit(code=1)

    output.mkdir(parents=True, exist_ok=True)
    for sample in candidates:
        dest = output / sample.path.name
        if mode == "copy":
            shutil.copy2(sample.path, dest)
        elif mode == "symlink":
            try:
                os.symlink(sample.path, dest)
            except OSError:
                shutil.copy2(sample.path, dest)
        else:
            console.print(f"[red]Unknown mode: {mode}[/red]")
            raise typer.Exit(code=1)


@app.command()
def stats(
    dataset_path: Annotated[Path, typer.Argument(help="Path to dataset directory")],
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: text|json|html"),
    ] = "text",
    chart: Annotated[
        bool,
        typer.Option("--chart", help="Generate charts (requires HTML format)"),
    ] = False,
) -> None:
    """Generate statistics and distribution analysis."""
    console.print(f"[bold cyan]Generating statistics for:[/bold cyan] {dataset_path}")
    console.print(f"[bold]Format:[/bold] {format}")

    if chart:
        console.print("[bold]Charts enabled[/bold]")

    config = AyaseConfig.load()
    module_names = _select_modules(quick=True, deep=False, config=config)
    modules = _instantiate_modules(module_names, config)
    pipeline = Pipeline(modules)
    samples = _iter_dataset_samples(dataset_path, include_videos=True, include_images=True)
    _run_pipeline(pipeline, samples)

    if format == "json":
        console.print_json(data=pipeline.stats.model_dump())
        return
    if format == "html":
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as tmp:
            tmp_path = Path(tmp.name)
        try:
            pipeline.export_report(tmp_path, format="html")
            console.print(tmp_path.read_text(encoding="utf-8"))
        finally:
            tmp_path.unlink(missing_ok=True)
        return
    if format != "text":
        console.print(f"[red]Unknown format: {format}[/red]")
        raise typer.Exit(code=1)

    stats = pipeline.stats
    console.print(f"Total samples: {stats.total_samples}")
    console.print(f"Valid samples: {stats.valid_samples}")
    console.print(f"Invalid samples: {stats.invalid_samples}")


# Modules subcommand
modules_app = typer.Typer(help="Inspect available pipeline modules and plugins")
app.add_typer(modules_app, name="modules")


@modules_app.command("list")
def modules_list() -> None:
    """List all discovered pipeline modules (built-in + plugins)."""
    config = AyaseConfig.load()
    _discover_all_modules(config)
    all_modules = ModuleRegistry.list_modules()

    if not all_modules:
        console.print("[yellow]No modules discovered.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title="Available Modules")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    for name in sorted(all_modules):
        table.add_row(name, all_modules[name])
    console.print(table)
    console.print(f"\n[dim]{len(all_modules)} module(s) total[/dim]")


@modules_app.command("check")
def modules_check() -> None:
    """Verify import/readiness using discovery and declared package requirements."""
    config = AyaseConfig.load()
    _discover_all_modules(config)
    all_modules = ModuleRegistry.list_modules()
    readiness = ModuleRegistry.readiness_report()

    if not all_modules and not readiness:
        console.print("[yellow]No modules discovered.[/yellow]")
        raise typer.Exit(code=0)

    errors = 0
    for name, info in sorted(readiness.items()):
        if info.get("status") != "ready":
            error = info.get("error") or "import failed"
            console.print(f"  [red]FAIL[/red] {name}: {error}")
            errors += 1

    for name in sorted(all_modules):
        cls = ModuleRegistry.get_module(name)
        module = None
        try:
            module = cls(
                config={
                    "models_dir": str(config.general.models_dir),
                    "parallel_jobs": config.general.parallel_jobs,
                }
            )
            missing = module._check_required_packages()
            if missing:
                console.print(
                    f"  [red]FAIL[/red] {name}: missing declared packages: {', '.join(missing)}"
                )
                errors += 1
                continue
            module.on_mount()
            if not getattr(module, "_mounted", False):
                console.print(f"  [red]FAIL[/red] {name}: module did not mount successfully")
                errors += 1
                continue
            console.print(f"  [green]OK[/green]  {name}")
        except Exception as e:
            console.print(f"  [red]FAIL[/red] {name}: {e}")
            errors += 1
        finally:
            if module is not None:
                try:
                    module.on_dispose()
                except Exception as e:
                    console.print(f"  [red]FAIL[/red] {name}: cleanup failed: {e}")
                    errors += 1

    if errors:
        console.print(f"\n[red]{errors} module(s) failed to load.[/red]")
        raise typer.Exit(code=1)
    console.print(f"\n[green]All {len(all_modules)} module(s) loaded successfully.[/green]")


@modules_app.command("docs")
def modules_docs(
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file (default: METRICS.md)")] = None,
    run_tests: Annotated[bool, typer.Option("--run-tests/--no-tests", help="Run tests and show pass/fail status")] = True,
) -> None:
    """Generate METRICS.md with charts, test status, and version info.

    Single command to regenerate everything:
        ayase modules docs -o METRICS.md --run-tests
    """
    config = AyaseConfig.load()
    _discover_all_modules(config)

    from .metrics_doc import generate_metrics_doc

    if run_tests:
        console.print("[cyan]Running tests to collect status...[/cyan]")
    content = generate_metrics_doc(run_tests=run_tests)

    if output:
        output.write_text(content, encoding="utf-8")
        console.print(f"[green]Written to {output}[/green]")
        # List generated charts
        docs_dir = Path("docs")
        if docs_dir.exists():
            charts = [f for f in docs_dir.iterdir() if f.suffix == ".png"]
            if charts:
                console.print(f"[green]Generated {len(charts)} charts in docs/[/green]")
    else:
        print(content)


@modules_app.command("models")
def modules_models(
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file (default: stdout)")] = None,
) -> None:
    """Generate MODELS.md catalog of all ML models and weights."""
    config = AyaseConfig.load()
    _discover_all_modules(config)

    from .models_doc import generate_models_doc

    content = generate_models_doc()

    if output:
        output.write_text(content, encoding="utf-8")
        console.print(f"[green]Written to {output}[/green]")
    else:
        print(content)


@modules_app.command("sync-readme")
def modules_sync_readme(
    readme: Annotated[Path, typer.Option("--readme", "-r", help="README.md path")] = Path("README.md"),
) -> None:
    """Update module/field counts in README.md to match reality."""
    import re as _re

    config = AyaseConfig.load()
    _discover_all_modules(config)

    all_modules = ModuleRegistry.list_modules()
    total = len([n for n in all_modules if ModuleRegistry.get_module(n) is not None])

    from .models import QualityMetrics
    n_fields = len(QualityMetrics.model_fields)

    if not readme.exists():
        console.print(f"[red]{readme} not found[/red]")
        raise typer.Exit(code=1)

    text = readme.read_text(encoding="utf-8")
    new_text = _re.sub(
        r"\*\*\d+ modules\*\*,\s*\*\*\d+ quality metrics\*\*",
        f"**{total} modules**, **{n_fields} quality metrics**",
        text,
    )
    if new_text != text:
        readme.write_text(new_text, encoding="utf-8")
        console.print(f"[green]Updated README.md: {total} modules, {n_fields} fields[/green]")
    else:
        console.print(f"[green]README.md already up to date ({total} modules, {n_fields} fields)[/green]")


# Config subcommand
config_app = typer.Typer(help="Manage configuration")
app.add_typer(config_app, name="config")


@config_app.command("init")
def config_init() -> None:
    """Initialize default config."""
    console.print("[bold]Initializing default config...[/bold]")
    config = AyaseConfig.load()
    path = Path("ayase.toml")
    if path.exists():
        console.print("[yellow]ayase.toml already exists; not overwriting.[/yellow]")
        return
    config.save(path)
    console.print(f"[green]Wrote default config to {path}[/green]")


@config_app.command("show")
def config_show() -> None:
    """Show current config."""
    console.print("[bold]Current configuration:[/bold]")
    config = AyaseConfig.load()
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    for key, value in config.model_dump().items():
        table.add_row(str(key), json.dumps(value, default=str))
    console.print(table)


@config_app.command("edit")
def config_edit() -> None:
    """Edit config in $EDITOR."""
    path = Path("ayase.toml")
    if not path.exists():
        AyaseConfig.load().save(path)
    editor = os.environ.get("EDITOR")
    if not editor:
        editor = "notepad" if os.name == "nt" else "vi"
    console.print(f"[bold]Opening config in editor ({editor})...[/bold]")
    subprocess.run([editor, str(path)])


@config_app.command("validate")
def config_validate() -> None:
    """Validate config file."""
    console.print("[bold green]Validating config file...[/bold green]")
    try:
        AyaseConfig.load()
        console.print("[green]Configuration is valid.[/green]")
    except Exception as e:
        console.print(f"[red]Configuration invalid: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def tui() -> None:
    """Launch the Terminal User Interface (TUI)."""
    try:
        from .tui import AyaseApp

        app = AyaseApp()
        app.run()
    except ImportError as e:
        console.print(f"[red]Error launching TUI: {e}[/red]")
        console.print(
            "[yellow]This install is missing required runtime dependencies. Reinstall with: pip install --upgrade --force-reinstall ayase[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")


if __name__ == "__main__":
    app()

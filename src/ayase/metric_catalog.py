"""Interactive catalog for discovering Ayase modules, metrics, models, and usage.

The catalog is built entirely from class metadata and source introspection. It
never instantiates a pipeline module, imports a model backend, or downloads
weights, so it is safe to use as an offline CLI help surface.
"""

from __future__ import annotations

import difflib
import inspect
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.table import Table

from .metrics_doc import (
    _CATEGORY_DISPLAY,
    _detect_backends,
    _detect_fallback_chain,
    _detect_dataset_fields_written,
    _detect_fields_written,
    _detect_gpu,
    _detect_packages,
    _detect_paper,
    _detect_speed_tier,
    _estimate_vram,
    _get_dataset_stats_fields,
    _get_quality_metrics_fields,
    _get_score_direction,
    _get_source,
)
from .models_doc import (
    _MODULE_MODEL_HINTS,
    _extract_clip_models,
    _extract_ffmpeg_models,
    _extract_hf_direct_urls,
    _extract_hf_models,
    _extract_pyiqa_metrics,
    _extract_required_files,
    _extract_torch_hub,
    _extract_torchvision_models,
    _get_module_source,
)
from .pipeline import ModuleRegistry, PipelineModule


@dataclass(frozen=True)
class MetricCatalogItem:
    """One sample- or dataset-level metric produced by a module."""

    name: str
    description: str
    scope: str
    category: str
    direction: str


@dataclass(frozen=True)
class ModelCatalogItem:
    """One model, weight file, or model-providing backend used by a module."""

    name: str
    source: str
    task: str = ""
    url: str = ""
    install: str = ""
    auto_download: bool = True


@dataclass(frozen=True)
class ModuleCatalogItem:
    """Structured help information for one registered pipeline module."""

    name: str
    description: str
    details: str
    input_type: str
    metrics: Tuple[MetricCatalogItem, ...]
    default_config: Dict[str, Any]
    models: Tuple[ModelCatalogItem, ...]
    packages: Tuple[str, ...]
    backends: Tuple[str, ...]
    fallback_chain: Tuple[str, ...]
    speed: str
    gpu: bool
    vram: str
    paper: str
    provisional: bool
    packaged: bool


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().casefold() not in {"0", "false", "no", "off"}
    return bool(value)


def _metric_description(
    field: str,
    detected: Dict[str, str],
    declared: Dict[str, str],
    schema: Dict[str, Dict[str, Any]],
) -> str:
    return (
        declared.get(field)
        or detected.get(field)
        or schema.get(field, {}).get("comment")
        or schema.get(field, {}).get("description")
        or ""
    )


@lru_cache(maxsize=1)
def _quality_schema() -> Dict[str, Dict[str, Any]]:
    return _get_quality_metrics_fields()


@lru_cache(maxsize=1)
def _dataset_schema() -> Dict[str, Dict[str, Any]]:
    return _get_dataset_stats_fields()


def _collect_metrics(cls: type, metadata: Dict[str, Any]) -> Tuple[MetricCatalogItem, ...]:
    quality_schema = _quality_schema()
    dataset_schema = _dataset_schema()
    declared_info = dict(getattr(cls, "metric_info", None) or {})
    declared_groups = dict(getattr(cls, "metric_groups", None) or {})

    sample_fields = dict(metadata.get("output_fields", {}))
    dataset_fields = dict(metadata.get("dataset_output_fields", {}))

    for field in declared_info:
        if field in quality_schema:
            sample_fields.setdefault(field, declared_info[field])
        elif field in dataset_schema:
            dataset_fields.setdefault(field, declared_info[field])
    for field in declared_groups:
        if field in quality_schema:
            sample_fields.setdefault(field, "")
        elif field in dataset_schema:
            dataset_fields.setdefault(field, "")

    metrics: List[MetricCatalogItem] = []
    for field in sample_fields:
        description = _metric_description(field, sample_fields, declared_info, quality_schema)
        group = declared_groups.get(field) or quality_schema.get(field, {}).get(
            "group", "other"
        )
        category = _CATEGORY_DISPLAY.get(group, group.replace("_", " ").title())
        metrics.append(
            MetricCatalogItem(
                name=field,
                description=description,
                scope="sample",
                category=category,
                direction=_get_score_direction(field, description),
            )
        )

    for field in dataset_fields:
        description = _metric_description(field, dataset_fields, declared_info, dataset_schema)
        metrics.append(
            MetricCatalogItem(
                name=field,
                description=description,
                scope="dataset",
                category="Dataset & Distribution",
                direction=_get_score_direction(field, description),
            )
        )
    return tuple(sorted(metrics, key=lambda metric: (metric.scope, metric.name)))


def _detect_input_type(cls: type, source: str) -> str:
    needs_reference = "reference_path" in source
    needs_caption = bool(re.search(r"caption.*\.text|\.caption", source[:3000]))
    video_only = bool(re.search(r"not\s+sample\.is_video", source))
    audio_module = bool(
        re.search(r"soundfile|librosa\.load|pesq|pystoi", source)
    ) or cls.name.startswith("audio_")
    batch_module = bool(
        re.search(r"post_process.*all_samples|batch", source[:500])
        and "def post_process" in source
    )

    if batch_module:
        input_type = "batch"
    elif audio_module:
        input_type = "audio"
    elif video_only:
        input_type = "vid"
    else:
        input_type = "img/vid"
    if needs_reference:
        input_type += " +ref"
    if needs_caption:
        input_type += " +cap"
    return input_type


def _module_details(cls: type, fallback: str) -> str:
    module = inspect.getmodule(cls)
    doc = inspect.getdoc(module) if module is not None else None
    if not doc:
        doc = inspect.getdoc(cls)
    return doc or fallback


def _model_key(item: ModelCatalogItem) -> Tuple[str, str]:
    if item.url:
        return "url", item.url
    return item.source, item.name.casefold()


def _collect_models(
    cls: type,
    metadata: Dict[str, Any],
    source: str,
) -> Tuple[ModelCatalogItem, ...]:
    found: Dict[Tuple[str, str], ModelCatalogItem] = {}

    def add(item: ModelCatalogItem) -> None:
        key = _model_key(item)
        previous = found.get(key)
        if previous is None:
            found[key] = item
            return
        found[key] = ModelCatalogItem(
            name=previous.name or item.name,
            source=previous.source or item.source,
            task=previous.task or item.task,
            url=previous.url or item.url,
            install=previous.install or item.install,
            auto_download=previous.auto_download and item.auto_download,
        )

    for declaration in metadata.get("models", []):
        model_id = str(declaration.get("id") or "").strip()
        if not model_id:
            continue
        source_type = str(declaration.get("type") or "other")
        url = str(declaration.get("url") or "")
        if source_type == "huggingface" and "/" in model_id and not url:
            url = f"https://huggingface.co/{model_id}"
        add(
            ModelCatalogItem(
                name=model_id,
                source=source_type,
                task=str(declaration.get("task") or ""),
                url=url,
                install=str(declaration.get("install") or ""),
                auto_download=_as_bool(declaration.get("auto_download"), True),
            )
        )

    default_config = metadata.get("default_config", {})
    for model_id in _extract_hf_models(source, default_config):
        add(
            ModelCatalogItem(
                name=model_id,
                source="huggingface",
                url=f"https://huggingface.co/{model_id}",
            )
        )
    for hint in _MODULE_MODEL_HINTS.get(cls.name, []):
        add(
            ModelCatalogItem(
                name=str(hint.get("name") or ""),
                source=str(hint.get("source") or "huggingface"),
                url=str(hint.get("url") or ""),
                install=str(hint.get("install") or ""),
                task=str(hint.get("notes") or ""),
            )
        )
    for metric in _extract_pyiqa_metrics(source):
        add(
            ModelCatalogItem(
                name=f"pyiqa/{metric}",
                source="pyiqa",
                install="pip install pyiqa",
            )
        )
    for repository in _extract_torch_hub(source):
        add(
            ModelCatalogItem(
                name=repository,
                source="torch_hub",
                install="pip install torch",
            )
        )
    for model_name in _extract_torchvision_models(source):
        add(
            ModelCatalogItem(
                name=f"torchvision/{model_name}",
                source="torchvision",
                install="pip install torchvision",
            )
        )
    for model_name in _extract_clip_models(source, default_config):
        add(
            ModelCatalogItem(
                name=f"CLIP {model_name}",
                source="clip",
                install="pip install open-clip-torch",
            )
        )
    for model_name in _extract_ffmpeg_models(source):
        add(
            ModelCatalogItem(
                name=f"ffmpeg/{model_name}",
                source="ffmpeg",
                install="Install FFmpeg with the required filter",
                auto_download=False,
            )
        )

    full_source = _get_module_source(cls)
    for repository, path in _extract_hf_direct_urls(full_source):
        add(
            ModelCatalogItem(
                name=path,
                source="huggingface_file",
                url=f"https://huggingface.co/{repository}/resolve/main/{path}",
            )
        )
    for filename, url in _extract_required_files(cls).items():
        add(
            ModelCatalogItem(
                name=filename,
                source="file",
                url=url,
            )
        )

    return tuple(sorted(found.values(), key=lambda item: (item.source, item.name.casefold())))


def _build_module_item(cls: type, detailed: bool) -> ModuleCatalogItem:
    source = _get_source(cls)
    if detailed:
        metadata = cls.get_metadata()
        backends = tuple(_detect_backends(source))
        packages = set(_detect_packages(source))
        packages.update(str(name) for name in getattr(cls, "required_packages", []) or [])
    else:
        sample_fields = {field: "" for field in _detect_fields_written(source)}
        dataset_fields = {
            field: "" for field in _detect_dataset_fields_written(source)
        }
        metadata = {
            "description": cls.description,
            "input_type": _detect_input_type(cls, source),
            "output_fields": sample_fields,
            "dataset_output_fields": dataset_fields,
            "default_config": {},
            "models": [],
        }
        backends = ()
        packages = set()
    return ModuleCatalogItem(
        name=cls.name,
        description=str(metadata.get("description") or cls.description),
        details=_module_details(cls, cls.description) if detailed else cls.description,
        input_type=str(metadata.get("input_type") or "unknown"),
        metrics=_collect_metrics(cls, metadata),
        default_config=dict(metadata.get("default_config") or {}),
        models=_collect_models(cls, metadata, source) if detailed else (),
        packages=tuple(sorted(packages)),
        backends=backends,
        fallback_chain=tuple(_detect_fallback_chain(source)) if detailed else (),
        speed=_detect_speed_tier(source, list(backends)) if detailed else "unknown",
        gpu=_detect_gpu(source) if detailed else False,
        vram=(_estimate_vram(source) or "unknown") if detailed else "unknown",
        paper=(_detect_paper(cls) or "") if detailed else "",
        provisional=bool(getattr(cls, "provisional", False)),
        packaged=ModuleRegistry.is_packaged_module(cls),
    )


class MetricCatalog:
    """Searchable snapshot of registered Ayase module metadata."""

    def __init__(self, modules: Sequence[ModuleCatalogItem]):
        self.modules = tuple(sorted(modules, key=lambda item: item.name))
        self._module_by_normalized = {
            _normalize_name(module.name): module for module in self.modules
        }
        metric_map: Dict[str, List[Tuple[ModuleCatalogItem, MetricCatalogItem]]] = {}
        for module in self.modules:
            for metric in module.metrics:
                metric_map.setdefault(metric.name, []).append((module, metric))
        self.metrics = {
            name: tuple(sorted(providers, key=lambda pair: pair[0].name))
            for name, providers in sorted(metric_map.items())
        }
        self._metric_by_normalized = {
            _normalize_name(name): name for name in self.metrics
        }

    def module(self, name: str) -> Optional[ModuleCatalogItem]:
        return self._module_by_normalized.get(_normalize_name(name))

    def metric(
        self, name: str
    ) -> Tuple[Tuple[ModuleCatalogItem, MetricCatalogItem], ...]:
        canonical = self._metric_by_normalized.get(_normalize_name(name))
        return self.metrics.get(canonical, ()) if canonical else ()

    def suggestions(self, query: str, limit: int = 6) -> Tuple[str, ...]:
        names = [module.name for module in self.modules] + list(self.metrics)
        normalized = _normalize_name(query)
        substring = [name for name in names if normalized in _normalize_name(name)]
        close = difflib.get_close_matches(query, names, n=limit, cutoff=0.45)
        ordered: List[str] = []
        for name in substring + close:
            if name not in ordered:
                ordered.append(name)
        return tuple(ordered[:limit])


def build_metric_catalog(
    module_names: Optional[Iterable[str]] = None,
    *,
    detailed: bool = True,
) -> MetricCatalog:
    """Build catalog entries for registered modules without instantiating them."""

    names = module_names or ModuleRegistry.list_modules().keys()
    modules: List[ModuleCatalogItem] = []
    for name in names:
        cls = ModuleRegistry.get_module(name)
        if cls is None or cls.name == "unnamed_module":
            continue
        modules.append(_build_module_item(cls, detailed=detailed))
    return MetricCatalog(modules)


def _render_metric_list(console: Console, catalog: MetricCatalog) -> None:
    table = Table(title="Available Ayase Metrics")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Module(s)", style="green")
    table.add_column("Scope")
    table.add_column("Input")
    table.add_column("Direction")
    table.add_column("Status")
    table.add_column("Description")

    for metric_name, providers in catalog.metrics.items():
        modules = ", ".join(module.name for module, _ in providers)
        scopes = ", ".join(sorted({metric.scope for _, metric in providers}))
        inputs = ", ".join(sorted({module.input_type for module, _ in providers}))
        directions = ", ".join(sorted({metric.direction for _, metric in providers}))
        statuses = {
            "experimental" if module.provisional else "ready" for module, _ in providers
        }
        description = next(
            (metric.description for _, metric in providers if metric.description), ""
        )
        table.add_row(
            metric_name,
            modules,
            scopes,
            inputs,
            directions,
            ", ".join(sorted(statuses)),
            description,
        )
    console.print(table)
    console.print(
        f"\n[dim]{len(catalog.metrics)} metric field(s) from "
        f"{len(catalog.modules)} module(s). Use `ayase help <metric-or-module>` "
        "for full details.[/dim]"
    )


def _render_module(
    console: Console,
    module: ModuleCatalogItem,
    selected_metric: Optional[str] = None,
) -> None:
    status = "experimental" if module.provisional else "ready"
    console.print(f"\n[bold cyan]{module.name}[/bold cyan] — {module.description}")

    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column("Property", style="bold")
    summary.add_column("Value")
    summary.add_row("Status", status)
    summary.add_row("Input", module.input_type)
    summary.add_row("Speed", module.speed)
    summary.add_row("GPU", "yes" if module.gpu else "no")
    summary.add_row("Estimated VRAM", module.vram)
    if module.paper:
        summary.add_row("Paper", module.paper)
    if module.backends:
        summary.add_row("Backends", ", ".join(module.backends))
    if module.fallback_chain:
        summary.add_row("Fallback chain", " → ".join(module.fallback_chain))
    if module.packages:
        summary.add_row("Dependencies", ", ".join(module.packages))
    console.print(summary)

    if module.details and module.details != module.description:
        console.print("\n[bold]What it does[/bold]")
        console.print(module.details)

    console.print("\n[bold]Output metrics[/bold]")
    metrics_table = Table()
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Scope")
    metrics_table.add_column("Category")
    metrics_table.add_column("Direction")
    metrics_table.add_column("Description")
    for metric in module.metrics:
        style = "bold cyan" if selected_metric == metric.name else "cyan"
        metrics_table.add_row(
            f"[{style}]{metric.name}[/]",
            metric.scope,
            metric.category,
            metric.direction,
            metric.description,
        )
    if module.metrics:
        console.print(metrics_table)
    else:
        console.print("[dim]No numeric QualityMetrics/DatasetStats fields declared.[/dim]")

    console.print("\n[bold]Models and weights[/bold]")
    if module.models:
        model_table = Table()
        model_table.add_column("Model / asset", style="magenta", overflow="fold")
        model_table.add_column("Source")
        model_table.add_column("Auto-download")
        model_table.add_column("Purpose")
        model_table.add_column("URL / install", overflow="fold")
        for model in module.models:
            location = model.url or model.install
            model_table.add_row(
                model.name,
                model.source,
                "yes" if model.auto_download else "no",
                model.task,
                location,
            )
        console.print(model_table)
    else:
        console.print("[dim]No external model or weight asset detected.[/dim]")

    console.print("\n[bold]Default configuration[/bold]")
    if module.default_config:
        config_table = Table()
        config_table.add_column("Key", style="yellow")
        config_table.add_column("Default")
        for key, value in sorted(module.default_config.items()):
            config_table.add_row(key, json.dumps(value, ensure_ascii=False, default=str))
        console.print(config_table)
    else:
        console.print("[dim]No module-specific configuration.[/dim]")

    placeholder = "DATASET_PATH" if any(
        metric.scope == "dataset" for metric in module.metrics
    ) else "MEDIA_PATH"
    profile = {
        "name": f"{module.name}_evaluation",
        "modules": [module.name],
        "module_config": {module.name: {}},
    }
    console.print("\n[bold]How to use[/bold]")
    console.print(f"  ayase scan {placeholder} --modules {module.name}")
    console.print(f"  ayase run {placeholder} --pipeline {module.name}")
    console.print("\n[bold]Profile example[/bold]")
    console.print(json.dumps(profile, indent=2, ensure_ascii=False))


def render_metric_help(
    console: Console,
    catalog: MetricCatalog,
    query: Optional[str] = None,
) -> bool:
    """Render list or detail help; return False when the query is unknown."""

    if not query:
        _render_metric_list(console, catalog)
        return True

    module = catalog.module(query)
    if module is not None:
        _render_module(console, module)
        return True

    providers = catalog.metric(query)
    if providers:
        metric_name = providers[0][1].name
        console.print(
            f"[bold cyan]Metric {metric_name}[/bold cyan] is produced by "
            f"{len(providers)} module(s)."
        )
        for owner, _ in providers:
            _render_module(console, owner, selected_metric=metric_name)
        return True

    console.print(f"[red]Unknown metric or module: {query}[/red]")
    suggestions = catalog.suggestions(query)
    if suggestions:
        console.print("[yellow]Did you mean:[/yellow] " + ", ".join(suggestions))
    return False

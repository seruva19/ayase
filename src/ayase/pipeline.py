import importlib
import importlib.util
import inspect
import json
import logging
import pkgutil
import sys
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Type, Any, Union

from .models import Sample, DatasetStats

logger = logging.getLogger(__name__)


class PipelineModule(ABC):
    """Base class for all quality assessment modules.

    Supports a ``test_mode`` flag that forces heuristic-only backends,
    skipping heavy ML model downloads. Activate via:

    - ``Module({"test_mode": True})`` — per-instance
    - ``AYASE_TEST_MODE=1`` environment variable — global
    - ``PipelineModule.set_test_mode(True)`` — class-level toggle

    In test mode, ``setup()`` is still called but modules should skip
    ML model loading when ``self.test_mode`` is True and fall through
    to their heuristic backend instead.
    """

    name: str = "unnamed_module"
    description: str = "No description provided"
    default_config: Dict[str, Any] = {}
    required_packages: List[str] = []
    required_files: Dict[str, str] = {}

    _global_test_mode: bool = False

    def __init__(self, config: Dict[str, Any] = None):
        self.config = {**self.default_config, **(config or {})}
        self.pipeline = None  # Will be set by Pipeline during initialization
        self._mounted = False

    @property
    def test_mode(self) -> bool:
        """Whether this module should skip ML model loading.

        True if any of: config ``test_mode``, env ``AYASE_TEST_MODE=1``,
        or class-level ``set_test_mode(True)`` is active.
        """
        import os
        return (
            self.config.get("test_mode", False)
            or self._global_test_mode
            or os.environ.get("AYASE_TEST_MODE", "") == "1"
        )

    @classmethod
    def set_test_mode(cls, enabled: bool = True) -> None:
        """Enable/disable test mode globally for all modules."""
        cls._global_test_mode = enabled

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name != "unnamed_module":
            ModuleRegistry.register(cls)

    @abstractmethod
    def process(self, sample: Sample) -> Sample:
        """Process a single sample and update its metrics/issues.

        Args:
            sample: The sample to process

        Returns:
            The updated sample
        """
        raise NotImplementedError("PipelineModule subclasses must implement process().")

    def on_mount(self) -> None:
        """Called when the module is loaded/initialized. Use for loading models/weights.

        In test mode, ``setup()`` is skipped entirely — modules stay in
        their default state with ``_ml_available = False``, which makes
        ``process()`` use the heuristic fallback (or return the sample
        unchanged). This avoids all heavy model downloads and GPU usage.
        """
        if self.test_mode:
            self._mounted = True
            return
        missing = self._check_required_packages()
        if missing:
            logger.warning(f"Missing packages for {self.name}: {', '.join(missing)}")
            return
        self._ensure_required_files()
        self.setup()
        self._mounted = True

    def on_execute(self) -> None:
        """Called before the pipeline starts processing samples."""
        return None

    def on_dispose(self) -> None:
        """Called when the pipeline finishes processing all samples. Use for cleanup."""
        # Backward compatibility for existing teardown()
        self.teardown()

    def setup(self) -> None:
        """Load ML models and weights. Override in subclasses.

        In test mode, this is automatically skipped by ``on_mount()``.
        If called directly (e.g. from tests), subclasses should handle
        gracefully when ML packages are unavailable.
        """
        return None

    def teardown(self) -> None:
        """Deprecated: Use on_dispose instead."""
        return None

    def post_process(self, all_samples: List[Sample]) -> None:
        """Called after all samples are processed. Use for cross-sample analysis."""
        return None

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Introspect module source to extract metadata without instantiation.

        Returns dict with: name, description, input_type, output_fields,
        default_config.  All inferred from existing code — no duplication.
        """
        import re as _re
        from .models import QualityMetrics

        # Field descriptions from QualityMetrics (source comments + pydantic fields)
        field_descs: Dict[str, str] = {}
        try:
            # Use pydantic model_fields for reliable field enumeration
            for fname in QualityMetrics.model_fields:
                field_descs[fname] = ""
            # Enrich with inline comments from source
            src_models = inspect.getsource(QualityMetrics)
            for m in _re.finditer(
                r"(\w+):\s*Optional\[.*?#\s*(.*)", src_models
            ):
                if m.group(1) in field_descs:
                    field_descs[m.group(1)] = m.group(2).strip()
        except (TypeError, OSError):
            pass

        # Source of the entire module file (not just the class)
        try:
            module_file = inspect.getfile(cls)
            with open(module_file, "r", encoding="utf-8", errors="replace") as _f:
                src = _f.read()
        except (TypeError, OSError):
            try:
                src = inspect.getsource(cls)
            except (TypeError, OSError):
                src = ""

        # Input type: infer from process() checks
        needs_ref = "reference_path" in src
        needs_cap = bool(
            _re.search(r"caption.*\.text|\.caption", src[:3000])
        )
        video_only = bool(_re.search(r"not\s+sample\.is_video", src))
        audio_module = bool(
            _re.search(r"soundfile|librosa\.load|pesq|pystoi", src)
        ) or cls.name.startswith("audio_")
        batch_module = bool(
            _re.search(r"post_process.*all_samples|batch", src[:500])
            and "def post_process" in src
        )

        if batch_module:
            input_type = "batch"
        elif audio_module:
            input_type = "audio"
        elif video_only:
            input_type = "vid"
        else:
            input_type = "img/vid"

        if needs_ref:
            input_type += " +ref"
        if needs_cap:
            input_type += " +cap"

        # Output fields: find quality_metrics.FIELD = ... assignments
        outputs: Dict[str, str] = {}
        for m in _re.finditer(r"quality_metrics\.(\w+)\s*=", src):
            field = m.group(1)
            if field not in outputs and field in field_descs:
                outputs[field] = field_descs[field]

        return {
            "name": cls.name,
            "description": cls.description,
            "input_type": input_type,
            "output_fields": outputs,
            "default_config": dict(cls.default_config) if cls.default_config else {},
        }

    def _check_required_packages(self) -> List[str]:
        required = []
        if isinstance(self.required_packages, list):
            required.extend(self.required_packages)
        config_required = self.config.get("required_packages")
        if isinstance(config_required, list):
            required.extend(config_required)
        missing = []
        for pkg in required:
            if importlib.util.find_spec(pkg) is None:
                missing.append(pkg)
        return missing

    def _ensure_required_files(self) -> None:
        required = {}
        if isinstance(self.required_files, dict):
            required.update(self.required_files)
        config_required = self.config.get("required_files")
        if isinstance(config_required, dict):
            required.update(config_required)
        model_urls = self.config.get("model_urls")
        if isinstance(model_urls, dict):
            required.update(model_urls)
        model_url = self.config.get("model_url")
        model_name = self.config.get("model_name")
        if isinstance(model_url, str) and isinstance(model_name, str):
            required[model_name] = model_url
        weights_url = self.config.get("weights_url")
        weights_name = self.config.get("weights_name")
        if isinstance(weights_url, str) and isinstance(weights_name, str):
            required[weights_name] = weights_url
        if not required:
            return
        models_dir = self.config.get("models_dir") or Path("models")
        models_dir = Path(models_dir)
        target_dir = models_dir / self.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename, url in required.items():
            if not url:
                continue
            target_path = target_dir / filename
            if target_path.exists():
                continue
            try:
                with urllib.request.urlopen(url, timeout=300) as resp:
                    with open(target_path, "wb") as f:
                        import shutil
                        shutil.copyfileobj(resp, f)
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")


class Pipeline:
    """Manages the execution of quality assessment modules."""

    def __init__(self, modules: List[PipelineModule]):
        self.modules = modules
        self.results: Dict[str, Sample] = {}
        self.stats = DatasetStats(total_samples=0, valid_samples=0, invalid_samples=0, total_size=0)
        self._batch_modules: List[PipelineModule] = []  # Modules that need batch processing
        # Maps stats field name -> (QualityMetrics field name, count)
        self._AVG_METRIC_MAP: Dict[str, str] = {
            "avg_technical_score": "technical_score",
            "avg_aesthetic_score": "aesthetic_score",
            "avg_motion_score": "motion_score",
        }
        self._metric_counts: Dict[str, int] = {k: 0 for k in self._AVG_METRIC_MAP}

        # Give modules access to pipeline for batch metrics
        for module in self.modules:
            module.pipeline = self

    def register_batch_module(self, module: PipelineModule) -> None:
        """Register a module that needs batch processing.

        Batch modules are called after all samples are processed via on_dispose().
        They can compute dataset-level metrics (e.g., FVD, KVD).

        Args:
            module: The module to register as batch processor
        """
        if module not in self._batch_modules:
            self._batch_modules.append(module)
            logger.debug(f"Registered batch module: {module.name}")

    def add_dataset_metric(self, metric_name: str, value: float) -> None:
        """Add a dataset-level metric to stats.

        Used by batch metric modules (FVD, KVD, etc.) to store their results.

        Args:
            metric_name: Name of the metric (e.g., "fvd", "kvd")
            value: Metric value
        """
        # Store in DatasetStats
        if hasattr(self.stats, metric_name):
            setattr(self.stats, metric_name, value)
            logger.info(f"Dataset metric {metric_name} = {value:.4f}")
        else:
            logger.warning(f"Unknown dataset metric: {metric_name}")

    def start(self) -> None:
        """Prepare all modules for execution."""
        for module in self.modules:
            try:
                if not getattr(module, "_mounted", False):
                    module.on_mount()
                    # on_mount() sets _mounted = True only when setup() succeeds.
                    # Do NOT force-set it here — modules with missing packages
                    # must remain unmounted so process_sample() skips them.
            except Exception as e:
                logger.error(f"Error in on_mount for module {module.name}: {e}")
        for module in self.modules:
            try:
                module.on_execute()
            except Exception as e:
                logger.error(f"Error in on_execute for module {module.name}: {e}")

    def stop(self) -> None:
        """Finalize and cleanup all modules."""
        all_samples = list(self.results.values())

        # Call post_process on all modules first
        for module in self.modules:
            try:
                module.post_process(all_samples)
            except Exception as e:
                logger.error(f"Error in post_process for module {module.name}: {e}")

        # Call on_dispose (this triggers batch metric computation)
        for module in self.modules:
            try:
                module.on_dispose()
            except Exception as e:
                logger.error(f"Error in on_dispose for module {module.name}: {e}")

        # Log batch metrics if any were computed
        if self._batch_modules:
            logger.info(f"Processed {len(self._batch_modules)} batch metric modules")
            for module in self._batch_modules:
                logger.debug(f"  - {module.name}")

    async def process_sample(self, sample: Sample) -> Sample:
        """Run all active modules on a sample."""

        # Check if we already have a result for this file in memory (from load_state)
        str_path = str(sample.path)
        if str_path in self.results:
            # You might want to check if the file has changed (mtime), but for now assume valid
            return self.results[str_path]

        for module in self.modules:
            if not getattr(module, "_mounted", False):
                continue
            try:
                sample = module.process(sample)
            except Exception as e:
                logger.error(f"Error in module {module.name} for {sample.path}: {e}")

        # Cache the result
        self.results[str_path] = sample

        # Update stats
        self.stats.total_samples += 1
        if sample.is_valid:
            self.stats.valid_samples += 1
        else:
            self.stats.invalid_samples += 1

        size = 0
        if sample.video_metadata:
            size = sample.video_metadata.file_size
        elif sample.image_metadata:
            size = sample.image_metadata.file_size
        elif sample.path.exists():
            try:
                size = sample.path.stat().st_size
            except OSError:
                logger.debug(f"Failed to stat size for {sample.path}")
        self.stats.total_size += size

        # Update Aggregated Stats
        if sample.quality_metrics:
            qm = sample.quality_metrics

            # Running average update (generic over _AVG_METRIC_MAP)
            for stats_field, qm_field in self._AVG_METRIC_MAP.items():
                value = getattr(qm, qm_field, None)
                if value is not None:
                    self._metric_counts[stats_field] += 1
                    c = self._metric_counts[stats_field]
                    prev = getattr(self.stats, stats_field, None) or 0.0
                    setattr(self.stats, stats_field, (prev * (c - 1) + value) / c)

        # Update Issue Stats
        for issue in sample.validation_issues:
            # By Severity
            sev = issue.severity.value
            self.stats.severity_distribution[sev] = self.stats.severity_distribution.get(sev, 0) + 1

            # By Type (heuristic from message prefix or module name)
            # We don't have explicit type field yet, so use message start
            key = issue.message.split(":")[0] if ":" in issue.message else issue.message[:20]
            self.stats.issues_by_type[key] = self.stats.issues_by_type.get(key, 0) + 1

        return sample

    def export_report(self, path: Path, format: str = "json") -> None:
        """Export a detailed validation report.

        Args:
            path: Output file path
            format: 'json', 'csv', or 'html'
        """
        path = Path(path)
        if format == "json":
            with open(path, "w", encoding="utf-8") as f:
                data = {
                    "stats": self.stats.model_dump(),
                    "samples": [s.model_dump(mode="json") for s in self.results.values()],
                }
                json.dump(data, f, indent=2, default=str)

        elif format == "csv":
            import csv

            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(["Path", "Valid", "Issues", "Recommendations", "Technical Score"])

                for s in self.results.values():
                    issues_str = "; ".join([i.message for i in s.validation_issues])
                    recs_str = "; ".join(
                        [i.recommendation for i in s.validation_issues if i.recommendation]
                    )
                    score = s.quality_metrics.technical_score if s.quality_metrics else 0

                    writer.writerow([str(s.path), s.is_valid, issues_str, recs_str, f"{score:.2f}"])

        elif format == "html":
            # Simple HTML report
            html = [
                "<html><head><title>Ayase Validation Report</title>",
                "<style>body{font-family:sans-serif} .error{color:red} .warn{color:orange}</style>",
                "</head><body>",
                f"<h1>Validation Report</h1>",
                f"<p>Total: {self.stats.total_samples} | Valid: {self.stats.valid_samples} | Invalid: {self.stats.invalid_samples}</p>",
                "<table border='1'><tr><th>Path</th><th>Status</th><th>Issues</th><th>Recommendations</th></tr>",
            ]

            from html import escape as _esc

            for s in self.results.values():
                status_color = "green" if s.is_valid else "red"
                issues_html = (
                    "<ul>"
                    + "".join([f"<li>{_esc(i.message)}</li>" for i in s.validation_issues])
                    + "</ul>"
                )
                recs_html = (
                    "<ul>"
                    + "".join(
                        [
                            f"<li>{_esc(i.recommendation)}</li>"
                            for i in s.validation_issues
                            if i.recommendation
                        ]
                    )
                    + "</ul>"
                )

                html.append(
                    f"<tr><td>{_esc(s.path.name)}</td><td style='color:{status_color}'>{s.is_valid}</td><td>{issues_html}</td><td>{recs_html}</td></tr>"
                )

            html.append("</table></body></html>")

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(html))

    def save_state(self, path: Path) -> None:
        """Save current pipeline state to disk for resume."""
        try:
            data = {
                "results": {k: v.model_dump(mode="json") for k, v in self.results.items()},
                "stats": self.stats.model_dump(mode="json"),
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"State saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self, path: Path) -> None:
        """Load pipeline state from disk."""
        if not path.exists():
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            if "results" in data:
                for k, v in data["results"].items():
                    try:
                        self.results[k] = Sample.model_validate(v)
                    except Exception as e:
                        logger.warning(f"Failed to load sample {k}: {e}")

            if "stats" in data:
                self.stats = DatasetStats.model_validate(data["stats"])

            # Rebuild metric counters from loaded samples to keep running averages stable.
            self._metric_counts = {k: 0 for k in self._AVG_METRIC_MAP}
            for sample in self.results.values():
                qm = sample.quality_metrics
                if not qm:
                    continue
                for stats_field, qm_field in self._AVG_METRIC_MAP.items():
                    if getattr(qm, qm_field, None) is not None:
                        self._metric_counts[stats_field] += 1

            logger.info(f"State loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")


class ModuleRegistry:
    """Registry for discovering and loading modules."""

    _modules: Dict[str, Type[PipelineModule]] = {}
    _readiness: Dict[str, Dict[str, Optional[str]]] = {}

    @classmethod
    def register(cls, module_cls: Type[PipelineModule]) -> None:
        cls._modules[module_cls.name] = module_cls

    @classmethod
    def get_module(cls, name: str) -> Optional[Type[PipelineModule]]:
        return cls._modules.get(name)

    @classmethod
    def list_modules(cls) -> Dict[str, str]:
        """Return dict of name -> description."""
        return {name: cls._modules[name].description for name in cls._modules}

    @classmethod
    def _record_readiness(cls, label: str, ok: bool, error: Optional[str] = None) -> None:
        if label in cls._readiness:
            return
        cls._readiness[label] = {
            "status": "ready" if ok else "missing",
            "error": error,
        }

    @classmethod
    def readiness_report(cls) -> Dict[str, Dict[str, Optional[str]]]:
        return dict(cls._readiness)

    @classmethod
    def discover_modules(
        cls,
        package_path: str = "ayase.modules",
        plugin_paths: Optional[List[Path]] = None,
    ) -> None:
        """Dynamically discover modules in the package."""
        try:
            module = importlib.import_module(package_path)
            if hasattr(module, "__path__"):
                for _, name, _ in pkgutil.iter_modules(module.__path__):
                    try:
                        importlib.import_module(f"{package_path}.{name}")
                        cls._record_readiness(name, True)
                    except Exception as e:
                        cls._record_readiness(name, False, str(e))
                        logger.warning(f"Failed to import module {package_path}.{name}: {e}")
        except ImportError:
            logger.warning(f"Could not import modules from {package_path}")
        if plugin_paths:
            cls.discover_external_modules(plugin_paths)

    @classmethod
    def discover_external_modules(cls, plugin_paths: List[Path]) -> None:
        for folder in plugin_paths:
            try:
                if not folder.exists() or not folder.is_dir():
                    continue
                for file_path in folder.glob("*.py"):
                    if file_path.name.startswith("_"):
                        continue
                    module_name = f"ayase_ext_{file_path.stem}_{abs(hash(str(file_path)))}"
                    if module_name in sys.modules:
                        continue
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if not spec or not spec.loader:
                        cls._record_readiness(file_path.stem, False, "Invalid module spec")
                        continue
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    try:
                        spec.loader.exec_module(module)
                        cls._record_readiness(file_path.stem, True)
                    except Exception as e:
                        cls._record_readiness(file_path.stem, False, str(e))
                        logger.warning(f"Failed to import external module {file_path}: {e}")
            except Exception as e:
                cls._record_readiness(str(folder), False, str(e))
                logger.warning(f"Failed to scan plugin folder {folder}: {e}")


class AyasePipeline:
    """High-level entry point for running Ayase pipelines.

    Wraps config loading, module discovery, profile instantiation,
    scanning, and async processing into a single facade.

    Usage::

        ayase = AyasePipeline()                         # all defaults
        ayase = AyasePipeline(modules=["basic", "fast_vqa"])
        ayase = AyasePipeline(profile="my_profile.toml")
        ayase = AyasePipeline(config=AyaseConfig.load("ayase.toml"))

        results = ayase.run("path/to/dataset")
        ayase.export("report.json")
    """

    def __init__(
        self,
        *,
        config: Optional[Any] = None,
        profile: Optional[Union[Path, str, Dict[str, Any]]] = None,
        modules: Optional[List[str]] = None,
    ):
        from .config import AyaseConfig

        # Load config
        if config is None:
            self.config = AyaseConfig.load()
        elif isinstance(config, (str, Path)):
            self.config = AyaseConfig.load(Path(config))
        else:
            self.config = config

        # Discover all available modules
        ModuleRegistry.discover_modules(
            plugin_paths=self.config.pipeline.plugin_folders,
        )

        # Build module list from profile or explicit list
        if profile is not None:
            from .profile import instantiate_profile_modules

            self._modules = instantiate_profile_modules(profile, self.config)
        elif modules is not None:
            self._modules = self._build_modules(modules)
        elif self.config.pipeline.modules:
            self._modules = self._build_modules(self.config.pipeline.modules)
        else:
            self._modules = []

        self.pipeline = Pipeline(self._modules)

    def _build_modules(self, names: List[str]) -> List[PipelineModule]:
        result = []
        for name in names:
            cls = ModuleRegistry.get_module(name)
            if cls is None:
                raise ValueError(f"Unknown module: {name}")
            result.append(
                cls(
                    config={
                        "models_dir": str(self.config.general.models_dir),
                    }
                )
            )
        return result

    def run(
        self,
        dataset_path: Union[str, Path],
        *,
        samples: Optional[Iterable[Sample]] = None,
        recursive: bool = True,
    ) -> Dict[str, Sample]:
        """Scan a dataset and process all samples through the pipeline.

        Args:
            dataset_path: Path to the dataset directory.
            samples: Pre-built samples (skip scanning if provided).
            recursive: Whether to scan subdirectories.

        Returns:
            Dict mapping file paths to processed Sample objects.
        """
        import asyncio

        from .scanner import scan_dataset

        if samples is None:
            samples = scan_dataset(Path(dataset_path), recursive=recursive)

        self.pipeline.start()
        try:
            loop = asyncio.new_event_loop()
            try:
                for sample in samples:
                    loop.run_until_complete(self.pipeline.process_sample(sample))
            finally:
                loop.close()
        finally:
            self.pipeline.stop()

        return self.pipeline.results

    def export(self, path: Union[str, Path], format: str = "json") -> None:
        """Export the pipeline report."""
        self.pipeline.export_report(Path(path), format=format)

    @property
    def results(self) -> Dict[str, Sample]:
        """Access processed sample results."""
        return self.pipeline.results

    @property
    def stats(self) -> DatasetStats:
        """Access aggregated dataset statistics."""
        return self.pipeline.stats

import importlib
import importlib.util
import inspect
import json
import logging
import os
import pkgutil
import sys
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Type, Any, Union

from .models import Sample, DatasetStats

logger = logging.getLogger(__name__)


class PipelineModule(ABC):
    """Base class for all quality assessment modules.

    Supports a ``test_mode`` flag that skips ML model loading,
    allowing fast unit testing without GPU or large downloads.
    Activate via:

    - ``Module({"test_mode": True})`` — per-instance
    - ``AYASE_TEST_MODE=1`` environment variable — global
    - ``PipelineModule.set_test_mode(True)`` — class-level toggle

    In test mode, ``setup()`` returns early and modules leave
    ``_ml_available = False``.  ``process()`` then returns the
    sample unchanged (no metrics computed).
    """

    name: str = "unnamed_module"
    description: str = "No description provided"
    default_config: Dict[str, Any] = {}
    required_packages: List[str] = []
    required_files: Dict[str, str] = {}
    models: List[Dict[str, str]] = []
    metric_info: Dict[str, str] = {}

    _global_test_mode: bool = False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self.default_config, **(config or {})}
        self.pipeline: Optional["Pipeline"] = None
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

    def __init_subclass__(cls, **kwargs: Any) -> None:
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
        self._release_torch_resources()

    def _release_torch_resources(self) -> None:
        """Best-effort release of torch model attrs and CUDA memory.

        Drops any instance attribute that is a torch.nn.Module so the GC
        can reclaim weights, then empties the CUDA cache if torch is
        already loaded. Skipped silently when torch isn't imported.
        """
        torch_mod = sys.modules.get("torch")
        if torch_mod is None:
            return
        try:
            nn_module_cls = torch_mod.nn.Module
        except AttributeError:
            return
        for attr_name, attr_val in list(self.__dict__.items()):
            if isinstance(attr_val, nn_module_cls):
                setattr(self, attr_name, None)
        try:
            if torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
        except Exception:
            pass

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
        from .models import QualityMetrics, DatasetStats

        # Field descriptions from QualityMetrics / DatasetStats
        # (source comments + pydantic fields).
        field_descs: Dict[str, str] = {}
        dataset_field_descs: Dict[str, str] = {}
        try:
            # Use pydantic model_fields for reliable field enumeration
            for fname in QualityMetrics.model_fields:
                field_descs[fname] = ""
            for fname in DatasetStats.model_fields:
                dataset_field_descs[fname] = ""
            # Enrich with inline comments from source
            src_models = inspect.getsource(QualityMetrics)
            for m in _re.finditer(
                r"(\w+):\s*Optional\[.*?#\s*(.*)", src_models
            ):
                if m.group(1) in field_descs:
                    field_descs[m.group(1)] = m.group(2).strip()
            src_stats = inspect.getsource(DatasetStats)
            for m in _re.finditer(
                r"(\w+):\s*Optional\[.*?#\s*(.*)", src_stats
            ):
                if m.group(1) in dataset_field_descs:
                    dataset_field_descs[m.group(1)] = m.group(2).strip()
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
        # Pattern 1: quality_metrics.FIELD =
        for m in _re.finditer(r"quality_metrics\.(\w+)\s*=", src):
            field = m.group(1)
            if field not in outputs and field in field_descs:
                outputs[field] = field_descs[field]
        # Pattern 2: metric_field = "FIELD" (base class auto-assignment)
        for m in _re.finditer(r'metric_field\s*=\s*["\'](\w+)["\']', src):
            field = m.group(1)
            if field not in outputs and field in field_descs:
                outputs[field] = field_descs[field]
        # Pattern 3: qm.FIELD = (variable alias for quality_metrics)
        if _re.search(r"\bqm\s*=\s*\w*\.?quality_metrics", src):
            for m in _re.finditer(r"\bqm\.(\w+)\s*=", src):
                field = m.group(1)
                if field not in outputs and field in field_descs:
                    outputs[field] = field_descs[field]

        dataset_outputs: Dict[str, str] = {}
        for m in _re.finditer(r'add_dataset_metric\(\s*["\'](\w+)["\']', src):
            field = m.group(1)
            if field not in dataset_outputs and field in dataset_field_descs:
                dataset_outputs[field] = dataset_field_descs[field]

        return {
            "name": cls.name,
            "description": cls.description,
            "input_type": input_type,
            "output_fields": outputs,
            "dataset_output_fields": dataset_outputs,
            "default_config": dict(cls.default_config) if cls.default_config else {},
            "models": list(cls.models) if cls.models else [],
            "metric_info": dict(cls.metric_info) if cls.metric_info else {},
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
        target_dir = (Path(models_dir) / self.name).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename, url in required.items():
            if not url:
                continue
            try:
                target_path = (target_dir / filename).resolve()
                target_path.relative_to(target_dir)
            except ValueError:
                logger.warning("Skipping unsafe required file path for %s: %s", self.name, filename)
                continue
            if target_path.exists():
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = target_path.with_suffix(target_path.suffix + ".part")
            try:
                with urllib.request.urlopen(url, timeout=300) as resp, open(tmp_path, "wb") as f:
                    import shutil

                    shutil.copyfileobj(resp, f)
                tmp_path.replace(target_path)
            except Exception as e:
                tmp_path.unlink(missing_ok=True)
                logger.warning(f"Failed to download {url}: {e}")


class Pipeline:
    """Manages the execution of quality assessment modules."""

    _REBUILT_STATS_FIELDS = {
        "total_samples",
        "valid_samples",
        "invalid_samples",
        "total_size",
        "avg_technical_score",
        "avg_aesthetic_score",
        "avg_motion_score",
        "issues_by_type",
        "severity_distribution",
    }
    _RUNTIME_CONFIG_KEYS = {"models_dir", "parallel_jobs"}

    def __init__(self, modules: List[PipelineModule]):
        self.modules = modules
        self.results: Dict[str, Sample] = {}
        self._result_signatures: Dict[str, tuple[object, ...]] = {}
        self._result_manifests: Dict[str, Dict[str, Any]] = {}
        self.stats = DatasetStats(total_samples=0, valid_samples=0, invalid_samples=0, total_size=0)
        self._batch_modules: List[PipelineModule] = []  # Modules that need batch processing
        self._hooks: Dict[str, Dict[str, Callable[[Sample], Sample]]] = {}
        # Maps stats field name -> (QualityMetrics field name, count)
        self._AVG_METRIC_MAP: Dict[str, str] = {
            "avg_technical_score": "technical_score",
            "avg_aesthetic_score": "aesthetic_score",
            "avg_motion_score": "motion_score",
        }
        self._metric_counts: Dict[str, int] = {k: 0 for k in self._AVG_METRIC_MAP}
        self._start_needs_reset = False

        # Give modules access to pipeline for batch metrics
        for module in self.modules:
            module.pipeline = self

    @staticmethod
    def _sample_cache_signature(sample: Sample) -> tuple[object, ...]:
        """Return the parts of a sample that materially affect processing output."""
        caption = sample.caption
        return (
            str(sample.path),
            sample.is_video,
            str(sample.reference_path) if sample.reference_path else None,
            caption.text if caption else None,
            str(caption.source_file) if caption and caption.source_file else None,
        )

    @classmethod
    def _normalize_fingerprint_value(cls, value: Any) -> Any:
        """Convert module config values into a stable, JSON-serializable structure."""
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): cls._normalize_fingerprint_value(sub_value)
                for key, sub_value in sorted(value.items(), key=lambda item: str(item[0]))
                if str(key) not in cls._RUNTIME_CONFIG_KEYS
            }
        if isinstance(value, (list, tuple)):
            return [cls._normalize_fingerprint_value(item) for item in value]
        if isinstance(value, set):
            return [
                cls._normalize_fingerprint_value(item)
                for item in sorted(value, key=lambda item: repr(item))
            ]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _pipeline_fingerprint(self) -> Dict[str, Any]:
        """Describe the active module stack for cache/state compatibility checks."""
        return {
            "modules": [
                {
                    "name": module.name,
                    "class": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
                    "config": self._normalize_fingerprint_value(module.config),
                    "test_mode": module.test_mode,
                }
                for module in self.modules
            ]
        }

    @staticmethod
    def _path_state_snapshot(path: Optional[Path]) -> Optional[Dict[str, Any]]:
        """Capture the current filesystem state needed to validate a cached file."""
        if path is None:
            return None
        try:
            stat = path.stat()
        except OSError:
            return {"path": str(path), "missing": True}
        return {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    @classmethod
    def _sample_state_manifest(cls, sample: Sample) -> Dict[str, Any]:
        """Build a manifest used to validate persisted cache entries on resume."""
        caption_source = sample.caption.source_file if sample.caption else None
        return {
            "media": cls._path_state_snapshot(sample.path),
            "reference": cls._path_state_snapshot(sample.reference_path),
            "caption_source": cls._path_state_snapshot(caption_source),
        }

    @staticmethod
    def _path_matches_snapshot(snapshot: Optional[Dict[str, Any]]) -> bool:
        """Return whether a file still matches its saved snapshot."""
        if snapshot is None:
            return True
        path = Path(snapshot["path"])
        missing = bool(snapshot.get("missing"))
        exists = path.exists()
        if missing:
            return not exists
        if not exists:
            return False
        try:
            stat = path.stat()
        except OSError:
            return False
        return (
            stat.st_size == snapshot.get("size")
            and stat.st_mtime_ns == snapshot.get("mtime_ns")
        )

    @classmethod
    def _sample_matches_manifest(cls, manifest: Dict[str, Any]) -> bool:
        """Return whether all files referenced by a manifest are unchanged."""
        return all(
            cls._path_matches_snapshot(manifest.get(field))
            for field in ("media", "reference", "caption_source")
        )

    @staticmethod
    def _sample_size(sample: Sample) -> int:
        """Return the best-known on-disk size for a sample."""
        if sample.video_metadata:
            return sample.video_metadata.file_size
        if sample.image_metadata:
            return sample.image_metadata.file_size
        if sample.path.exists():
            try:
                return sample.path.stat().st_size
            except OSError:
                logger.debug(f"Failed to stat size for {sample.path}")
        return 0

    def _reset_rebuilt_stats(self) -> None:
        """Reset sample-derived aggregate stats before rebuilding from results."""
        self.stats = DatasetStats(total_samples=0, valid_samples=0, invalid_samples=0, total_size=0)
        self._metric_counts = {k: 0 for k in self._AVG_METRIC_MAP}

    def _update_average_stat(self, stats_field: str, value: Optional[float], delta: int) -> None:
        """Apply or remove a sample value from a running average."""
        if value is None:
            return
        count = self._metric_counts[stats_field]
        prev_avg = getattr(self.stats, stats_field, None) or 0.0
        if delta > 0:
            new_count = count + 1
            new_avg = ((prev_avg * count) + value) / new_count
        else:
            if count == 0:
                return
            new_count = count - 1
            if new_count == 0:
                self._metric_counts[stats_field] = 0
                setattr(self.stats, stats_field, None)
                return
            new_avg = ((prev_avg * count) - value) / new_count
        self._metric_counts[stats_field] = new_count
        setattr(self.stats, stats_field, new_avg)

    @staticmethod
    def _update_issue_counter(counter: Dict[str, int], key: str, delta: int) -> None:
        """Increment or decrement a keyed counter, removing empty entries."""
        new_value = counter.get(key, 0) + delta
        if new_value > 0:
            counter[key] = new_value
        else:
            counter.pop(key, None)

    def _apply_sample_stats(self, sample: Sample, delta: int) -> None:
        """Apply or remove a sample's contribution from aggregate stats."""
        self.stats.total_samples += delta
        if sample.is_valid:
            self.stats.valid_samples += delta
        else:
            self.stats.invalid_samples += delta

        self.stats.total_size += self._sample_size(sample) * delta

        qm = sample.quality_metrics
        if qm:
            for stats_field, qm_field in self._AVG_METRIC_MAP.items():
                self._update_average_stat(stats_field, getattr(qm, qm_field, None), delta)

        for issue in sample.validation_issues:
            sev = issue.severity.value
            self._update_issue_counter(self.stats.severity_distribution, sev, delta)
            key = issue.message.split(":")[0] if ":" in issue.message else issue.message[:20]
            self._update_issue_counter(self.stats.issues_by_type, key, delta)

    def _store_result(
        self,
        key: str,
        sample: Sample,
        *,
        signature: tuple[object, ...],
        manifest: Dict[str, Any],
    ) -> None:
        """Store a processed sample while keeping aggregate stats consistent."""
        previous = self.results.get(key)
        if previous is not None:
            self._apply_sample_stats(previous, -1)
        self.results[key] = sample
        self._result_signatures[key] = signature
        self._result_manifests[key] = manifest
        self._apply_sample_stats(sample, 1)

    def _restore_saved_stats(self, saved_stats: DatasetStats) -> None:
        """Restore persisted dataset-level metrics that are not rebuilt from samples."""
        for field in DatasetStats.model_fields:
            if field in self._REBUILT_STATS_FIELDS:
                continue
            setattr(self.stats, field, getattr(saved_stats, field))

    def _clear_loaded_state(self) -> None:
        """Reset cached results and rebuilt stats before replacing state."""
        self.results = {}
        self._result_signatures = {}
        self._result_manifests = {}
        self._reset_rebuilt_stats()

    @staticmethod
    def _sample_matches_basic_cache(sample: Sample) -> bool:
        """Backward-compatible stale-cache validation for legacy state files."""
        try:
            stat = sample.path.stat()
        except OSError:
            return False
        cached_size = None
        if sample.video_metadata:
            cached_size = sample.video_metadata.file_size
        elif sample.image_metadata:
            cached_size = sample.image_metadata.file_size
        return cached_size is None or stat.st_size == cached_size

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

    def add_hook(
        self,
        module_name: str,
        *,
        before: Optional[Callable[[Sample], Sample]] = None,
        after: Optional[Callable[[Sample], Sample]] = None,
    ) -> None:
        """Register before/after hooks for a module.

        Hooks are called around ``module.process(sample)`` inside
        ``process_sample()``.  A *before* hook can modify the sample
        (e.g. condense a caption) before the module sees it; an *after*
        hook can restore the original state so subsequent modules are
        unaffected.

        Calling ``add_hook`` again for the same *module_name* replaces
        previously registered callbacks.

        Args:
            module_name: ``PipelineModule.name`` of the target module.
            before: ``(Sample) -> Sample`` called before ``process()``.
            after:  ``(Sample) -> Sample`` called after ``process()``.
        """
        entry: Dict[str, Callable[[Sample], Sample]] = {}
        if before is not None:
            entry["before"] = before
        if after is not None:
            entry["after"] = after
        if entry:
            self._hooks[module_name] = entry

    def add_dataset_metric(self, metric_name: str, value: Any) -> None:
        """Add a dataset-level metric to stats.

        Used by batch metric modules (FVD, KVD, etc.) to store their results.

        Args:
            metric_name: Name of the metric (e.g., "fvd", "kvd")
            value: Metric value
        """
        # Store in DatasetStats
        if hasattr(self.stats, metric_name):
            setattr(self.stats, metric_name, value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                logger.info("Dataset metric %s = %.4f", metric_name, value)
            else:
                logger.info("Dataset metric %s updated", metric_name)
        else:
            logger.warning(f"Unknown dataset metric: {metric_name}")

    def start(self) -> None:
        """Prepare all modules for execution."""
        if self._start_needs_reset:
            self._clear_loaded_state()
            self._start_needs_reset = False
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
        self._start_needs_reset = True

    def process_sample(self, sample: Sample) -> Sample:
        """Run all active modules on a sample."""

        # Check if we already have a result for this file in memory (from load_state)
        str_path = str(sample.path)
        signature = self._sample_cache_signature(sample)
        manifest = self._sample_state_manifest(sample)
        cached = self.results.get(str_path)
        if (
            cached is not None
            and self._result_signatures.get(str_path) == signature
            and self._result_manifests.get(str_path) == manifest
        ):
            return cached

        for module in self.modules:
            if not getattr(module, "_mounted", False):
                continue
            try:
                hooks = self._hooks.get(module.name)
                if hooks and "before" in hooks:
                    hooked = hooks["before"](sample)
                    if not isinstance(hooked, Sample):
                        logger.error(
                            "Before-hook for module %s returned %s for %s; "
                            "skipping module",
                            module.name,
                            type(hooked).__name__,
                            str_path,
                        )
                        continue
                    sample = hooked

                processed = module.process(sample)
                if not isinstance(processed, Sample):
                    logger.error(
                        "Module %s returned %s for %s; keeping previous sample",
                        module.name,
                        type(processed).__name__,
                        str_path,
                    )
                    continue
                sample = processed

                if hooks and "after" in hooks:
                    restored = hooks["after"](sample)
                    if not isinstance(restored, Sample):
                        logger.error(
                            "After-hook for module %s returned %s for %s; "
                            "keeping module output",
                            module.name,
                            type(restored).__name__,
                            str_path,
                        )
                        continue
                    sample = restored
            except Exception as e:
                sample_path = getattr(sample, "path", str_path)
                logger.error(f"Error in module {module.name} for {sample_path}: {e}")

        # Cache the result and keep aggregate stats in sync.
        self._store_result(str_path, sample, signature=signature, manifest=manifest)

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
                    score = (s.quality_metrics.technical_score if s.quality_metrics else None) or 0.0

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
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "pipeline_fingerprint": self._pipeline_fingerprint(),
                "results": {k: v.model_dump(mode="json") for k, v in self.results.items()},
                "stats": self.stats.model_dump(mode="json"),
                "cache_manifest": {
                    k: self._result_manifests.get(k) or self._sample_state_manifest(v)
                    for k, v in self.results.items()
                },
            }
            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                Path(tmp_path).replace(path)
            except Exception:
                Path(tmp_path).unlink(missing_ok=True)
                raise
            logger.info(f"State saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self, path: Path) -> None:
        """Load pipeline state from disk."""
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            saved_fingerprint = data.get("pipeline_fingerprint")
            current_fingerprint = self._pipeline_fingerprint()
            saved_stats = None
            if "stats" in data:
                saved_stats = DatasetStats.model_validate(data["stats"])
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return

        if not isinstance(saved_fingerprint, dict):
            self._clear_loaded_state()
            self._start_needs_reset = False
            logger.info("Skipping legacy state file without pipeline fingerprint: %s", path)
            return
        if saved_fingerprint != current_fingerprint:
            self._clear_loaded_state()
            self._start_needs_reset = False
            logger.info("Skipping state file with incompatible pipeline fingerprint: %s", path)
            return

        previous_results = self.results
        previous_signatures = self._result_signatures
        previous_manifests = self._result_manifests
        previous_stats = self.stats.model_copy(deep=True)
        previous_metric_counts = dict(self._metric_counts)
        previous_start_needs_reset = self._start_needs_reset
        self._clear_loaded_state()
        self._start_needs_reset = False
        try:
            results_data = data.get("results", {})
            manifests = data.get("cache_manifest", {})
            partial_restore = False

            if "results" in data:
                for k, v in results_data.items():
                    try:
                        sample = Sample.model_validate(v)
                        manifest = manifests.get(k) if isinstance(manifests, dict) else None
                        if isinstance(manifest, dict):
                            if not self._sample_matches_manifest(manifest):
                                logger.info(f"Skipping stale cache for {k} (state manifest changed)")
                                partial_restore = True
                                continue
                        elif not self._sample_matches_basic_cache(sample):
                            logger.info(f"Skipping stale cache for {k} (file changed or missing)")
                            partial_restore = True
                            continue
                        self.results[k] = sample
                        self._result_signatures[k] = self._sample_cache_signature(sample)
                        self._result_manifests[k] = (
                            manifest if isinstance(manifest, dict) else self._sample_state_manifest(sample)
                        )
                        self._apply_sample_stats(sample, 1)
                    except Exception as e:
                        partial_restore = True
                        logger.warning(f"Failed to load sample {k}: {e}")

            if saved_stats is not None and not partial_restore and len(self.results) == len(results_data):
                self._restore_saved_stats(saved_stats)

            logger.info(f"State loaded from {path}")
        except Exception as e:
            self.results = previous_results
            self._result_signatures = previous_signatures
            self._result_manifests = previous_manifests
            self.stats = previous_stats
            self._metric_counts = previous_metric_counts
            self._start_needs_reset = previous_start_needs_reset
            logger.error(f"Failed to load state: {e}")


class ModuleRegistry:
    """Registry for discovering and loading modules."""

    _modules: Dict[str, Type[PipelineModule]] = {}
    _readiness: Dict[str, Dict[str, Optional[str]]] = {}
    _external_plugin_labels: Dict[str, Set[str]] = {}
    _external_plugin_modules: Dict[str, str] = {}

    @classmethod
    def register(cls, module_cls: Type[PipelineModule]) -> None:
        existing = cls._modules.get(module_cls.name)
        if existing is not None and existing is not module_cls:
            raise ValueError(
                f"Duplicate module name '{module_cls.name}' for "
                f"{existing.__module__}.{existing.__name__} and "
                f"{module_cls.__module__}.{module_cls.__name__}"
            )
        cls._modules[module_cls.name] = module_cls

    @classmethod
    def get_module(cls, name: str) -> Optional[Type[PipelineModule]]:
        return cls._modules.get(name)

    @classmethod
    def is_packaged_module(cls, module_cls: Type[PipelineModule]) -> bool:
        """Return whether a registered module ships from ``ayase.modules``."""
        return getattr(module_cls, "__module__", "").startswith("ayase.modules.")

    @classmethod
    def list_modules(cls, packaged_only: bool = False) -> Dict[str, str]:
        """Return dict of name -> description, sorted by name for stable iteration."""
        return {
            name: cls._modules[name].description
            for name in sorted(cls._modules)
            if not packaged_only or cls.is_packaged_module(cls._modules[name])
        }

    @classmethod
    def _record_readiness(cls, label: str, ok: bool, error: Optional[str] = None) -> None:
        cls._readiness[label] = {
            "status": "ready" if ok else "missing",
            "error": error,
        }

    @classmethod
    def _rollback_partial_registration(
        cls,
        previous_modules: Dict[str, Type[PipelineModule]],
        module_name: str,
    ) -> None:
        """Remove any classes auto-registered by a module that failed to import."""
        for name, module_cls in list(cls._modules.items()):
            if previous_modules.get(name) is module_cls:
                continue
            if getattr(module_cls, "__module__", None) == module_name:
                cls._modules.pop(name, None)

    @classmethod
    def _remove_registered_classes_for_module(cls, module_name: str) -> None:
        """Unregister all module classes that came from a given imported module."""
        for name, module_cls in list(cls._modules.items()):
            if getattr(module_cls, "__module__", None) == module_name:
                cls._modules.pop(name, None)

    @classmethod
    def readiness_report(cls) -> Dict[str, Dict[str, Optional[str]]]:
        return dict(cls._readiness)

    @staticmethod
    def _external_plugin_label(file_path: Path) -> str:
        """Return a stable readiness label for an external plugin file."""
        return str(file_path.resolve())

    @classmethod
    def _update_external_plugin_labels(cls, folder: Path, labels: Set[str]) -> None:
        """Prune readiness entries for plugin files that no longer exist."""
        folder_key = str(folder.resolve())
        previous = cls._external_plugin_labels.get(folder_key, set())
        normalized = set(labels)
        if normalized:
            cls._external_plugin_labels[folder_key] = normalized
        else:
            cls._external_plugin_labels.pop(folder_key, None)

        other_labels: Set[str] = set()
        for other_key, other_set in cls._external_plugin_labels.items():
            if other_key != folder_key:
                other_labels.update(other_set)

        for label in previous - normalized:
            if label not in other_labels:
                cls._readiness.pop(label, None)

    @classmethod
    def _prune_external_plugin_modules(cls, folder: Path, current_file_keys: Set[str]) -> None:
        """Unload previously discovered plugin modules whose files disappeared."""
        folder_key = str(folder.resolve())
        previous_file_keys = {
            file_key
            for file_key in cls._external_plugin_modules
            if str(Path(file_key).parent) == folder_key
        }
        for file_key in previous_file_keys - current_file_keys:
            stale_module_name = cls._external_plugin_modules.pop(file_key, None)
            if stale_module_name is None:
                continue
            sys.modules.pop(stale_module_name, None)
            cls._remove_registered_classes_for_module(stale_module_name)

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
                    module_name = f"{package_path}.{name}"
                    previous_modules = dict(cls._modules)
                    try:
                        importlib.import_module(module_name)
                        cls._record_readiness(name, True)
                    except Exception as e:
                        sys.modules.pop(module_name, None)
                        cls._rollback_partial_registration(previous_modules, module_name)
                        cls._record_readiness(name, False, str(e))
                        logger.warning(f"Failed to import module {module_name}: {e}")
        except ImportError:
            logger.warning(f"Could not import modules from {package_path}")
        if plugin_paths:
            cls.discover_external_modules(plugin_paths)

    @classmethod
    def discover_external_modules(cls, plugin_paths: List[Path]) -> None:
        for folder in plugin_paths:
            current_labels: Set[str] = set()
            current_file_keys: Set[str] = set()
            try:
                if not folder.exists() or not folder.is_dir():
                    cls._prune_external_plugin_modules(folder, current_file_keys)
                    cls._readiness.pop(str(folder), None)
                    cls._update_external_plugin_labels(folder, current_labels)
                    continue
                cls._readiness.pop(str(folder), None)
                for file_path in folder.glob("*.py"):
                    if file_path.name.startswith("_"):
                        continue
                    file_key = str(file_path.resolve())
                    readiness_label = cls._external_plugin_label(file_path)
                    current_file_keys.add(file_key)
                    current_labels.add(readiness_label)
                    previous_module_name = cls._external_plugin_modules.get(file_key)
                    if previous_module_name is not None:
                        sys.modules.pop(previous_module_name, None)
                        cls._remove_registered_classes_for_module(previous_module_name)
                    module_name = f"ayase_ext_{file_path.stem}_{abs(hash(file_key))}"
                    cls._external_plugin_modules[file_key] = module_name
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if not spec or not spec.loader:
                        cls._record_readiness(readiness_label, False, "Invalid module spec")
                        continue
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    previous_modules = dict(cls._modules)
                    try:
                        source = file_path.read_text(encoding="utf-8")
                        exec(compile(source, str(file_path), "exec"), module.__dict__)
                        cls._record_readiness(readiness_label, True)
                    except Exception as e:
                        sys.modules.pop(module_name, None)
                        cls._rollback_partial_registration(previous_modules, module_name)
                        cls._record_readiness(readiness_label, False, str(e))
                        logger.warning(f"Failed to import external module {file_path}: {e}")
                cls._prune_external_plugin_modules(folder, current_file_keys)
                cls._update_external_plugin_labels(folder, current_labels)
            except Exception as e:
                cls._prune_external_plugin_modules(folder, current_file_keys)
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

        self.pipeline: Pipeline
        self._rebuild_pipeline()

    def _clone_modules(
        self,
        templates: Optional[List[PipelineModule]] = None,
    ) -> List[PipelineModule]:
        """Recreate module instances so each run starts from clean module state."""
        source = self._modules if templates is None else templates
        return [
            module.__class__(config=deepcopy(module.config))
            for module in source
        ]

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
                        "parallel_jobs": self.config.general.parallel_jobs,
                    }
                )
            )
        return result

    @staticmethod
    def _clone_pipeline_hooks(
        hooks: Dict[str, Dict[str, Callable[[Sample], Sample]]],
    ) -> Dict[str, Dict[str, Callable[[Sample], Sample]]]:
        """Copy pipeline hooks without sharing inner dicts."""
        return {module_name: dict(callbacks) for module_name, callbacks in hooks.items()}

    def _rebuild_pipeline(self, preserve_public_state: bool = False) -> Pipeline:
        """Create a fresh pipeline while keeping user-visible customizations intact."""
        module_templates = self._modules
        hooks: Dict[str, Dict[str, Callable[[Sample], Sample]]] = {}
        if preserve_public_state and hasattr(self, "pipeline"):
            module_templates = self.pipeline.modules
            hooks = self._clone_pipeline_hooks(self.pipeline._hooks)
        self.pipeline = Pipeline(self._clone_modules(module_templates))
        self.pipeline._hooks = hooks
        return self.pipeline

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
        from .scanner import scan_dataset

        if samples is None:
            samples = scan_dataset(Path(dataset_path), recursive=recursive)

        pipeline = self._rebuild_pipeline(preserve_public_state=True)
        pipeline.start()
        try:
            for sample in samples:
                pipeline.process_sample(sample)
        finally:
            pipeline.stop()

        return pipeline.results

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

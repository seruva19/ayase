import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import os
import pkgutil
import sys
import tempfile
from time import perf_counter
import urllib.request
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Set, Type, Union, cast

from .models import (
    Sample,
    DatasetStats,
    QualityMetrics,
    ValidationIssue,
    ValidationSeverity,
    move_lip_sync_legacy_aliases,
    observe_metric_writes,
)
from .runtime import (
    clone_frames,
    opt_in_all_provenance,
    pipeline_context,
    readonly_view,
    runtime_module_config,
)

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
    metric_groups: Dict[str, str] = {}

    # Field-level provenance declarations (see AGENTS.md and METRICS.md):
    #
    #   provenance — output field name -> class:
    #       "published" — computed per the published definition (same model /
    #                   weights / aggregation / protocol as the source);
    #       "adapted"   — published metric whose implementation deviates so the
    #                   numbers will not match (then ``deviations`` is required);
    #       "own"       — no published definition exists for this quantity;
    #       "utility"   — not a quality score (metadata, detectors, helpers).
    #   sources    — field -> primary source (paper citation + URL/DOI);
    #               required for every "published" field.
    #   deviations — field -> how the implementation deviates from the source;
    #               required for every "adapted" field.
    #
    # All three accept a ``"*"`` key as the per-module default for fields not
    # listed explicitly; ``provenance`` also accepts a plain string as shorthand
    # applying one class to every output field of the module. Modules with no
    # ``provenance`` at all are treated as unmarked (allowed to run, but the
    # provenance test suite requires declarations on packaged modules).
    provenance: Union[Dict[str, str], str] = {}
    sources: Dict[str, str] = {}
    deviations: Dict[str, str] = {}

    # True = no turnkey real backend in a standard install (uninstallable dep,
    # unreleased weights, needs training, or architecturally impossible). The
    # module stays registered and revivable but is excluded from documented
    # metric/module counts (README, METRICS.md, MODELS.md) and is instead shown
    # in an "External backend required — pending real backend" section. Flip to False the
    # moment a reachable real backend lands.
    requires_external_backend: bool = False

    # True = Ayase-defined construct without a published definition, slated for
    # removal in a future release. Instantiating the module emits a
    # DeprecationWarning; the module keeps working until removal.
    deprecated: bool = False

    _global_test_mode: bool = False

    # Declare a media type when source inference cannot identify the contract.
    input_type: ClassVar[Optional[str]] = None
    _metadata_field_descriptions: ClassVar[tuple[Dict[str, str], Dict[str, str]]]
    _metadata_cache: ClassVar[Dict[str, Any]]
    _metadata_cache_key: ClassVar[tuple[Any, ...]]
    _resolved_field_provenance: ClassVar[Dict[str, str]]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if getattr(type(self), "deprecated", False):
            warnings.warn(
                f"Module '{getattr(type(self), 'name', type(self).__name__)}' is "
                "an Ayase-defined construct without a published definition and "
                "is deprecated; it will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.config = self._merge_config(self.default_config, config)
        self.pipeline: Optional["Pipeline"] = None
        self._mounted = False

    @classmethod
    def _merge_config(
        cls,
        base: Optional[Dict[str, Any]],
        override: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Deep-merge *override* onto a deep copy of *base*.

        The class-level ``default_config`` must never be shared by reference:
        several modules keep nested dicts (e.g. ``"weights"``/``"dimensions"``)
        and an in-place mutation on one instance would otherwise poison every
        future instance and its config fingerprint. Nested dict overrides merge
        key-wise rather than replacing the whole sub-dict.
        """
        merged: Dict[str, Any] = deepcopy(base) if isinstance(base, dict) else {}
        if not override:
            return merged
        for key, value in override.items():
            existing = merged.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                merged[key] = cls._merge_config(existing, value)
            else:
                merged[key] = deepcopy(value)
        return merged

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

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        """Process multiple samples.

        Subclasses may override this for real batched inference. The default
        implementation preserves legacy single-sample behavior and catches
        per-sample failures so one bad item does not abort the whole batch.
        """
        processed_samples = []
        for sample in samples:
            try:
                processed = self.process(sample)
            except Exception as e:
                logger.error(f"Error in module {self.name} for {sample.path}: {e}")
                Pipeline._register_module_failure(sample, self.name, f"{type(e).__name__}: {e}")
                processed_samples.append(sample)
                continue
            if not isinstance(processed, Sample):
                logger.error(
                    "Module %s returned %s for %s; keeping previous sample",
                    self.name,
                    type(processed).__name__,
                    sample.path,
                )
                Pipeline._register_module_failure(
                    sample,
                    self.name,
                    f"returned {type(processed).__name__}, expected Sample",
                )
                processed_samples.append(sample)
                continue
            processed_samples.append(processed)
        return processed_samples

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
        default_config. Media types may be declared explicitly; other metadata
        is inferred from source and field declarations.
        """
        import re as _re
        from .models import QualityMetrics, DatasetStats

        cached_metadata = cls.__dict__.get("_metadata_cache")
        cache_key = (
            cls.name,
            cls.description,
            repr(cls.default_config),
            repr(cls.models),
            repr(cls.provenance),
            repr(cls.sources),
            repr(cls.deviations),
            repr(cls.metric_info),
            repr(cls.metric_groups),
            repr(getattr(cls, "metric_field", None)),
            repr(getattr(cls, "metric_field_name", None)),
            cls.requires_external_backend,
            cls.deprecated,
            cls.input_type,
        )
        if cached_metadata is not None and cls.__dict__.get("_metadata_cache_key") == cache_key:
            return deepcopy(cached_metadata)

        # Field descriptions from QualityMetrics / DatasetStats
        # (source comments + pydantic fields).
        cached_descriptions = PipelineModule.__dict__.get("_metadata_field_descriptions")
        if cached_descriptions is not None:
            field_descs, dataset_field_descs = deepcopy(cached_descriptions)
        else:
            field_descs = {}
            dataset_field_descs = {}
            # Use pydantic model_fields for reliable field enumeration
            for fname in QualityMetrics.model_fields:
                field_descs[fname] = ""
            for fname in DatasetStats.model_fields:
                dataset_field_descs[fname] = ""
            # Enrich with inline comments from source
            src_models = inspect.getsource(QualityMetrics)
            for m in _re.finditer(r"(\w+):\s*Optional\[.*?#\s*(.*)", src_models):
                if m.group(1) in field_descs:
                    field_descs[m.group(1)] = m.group(2).strip()
            src_stats = inspect.getsource(DatasetStats)
            for m in _re.finditer(r"(\w+):\s*Optional\[.*?#\s*(.*)", src_stats):
                if m.group(1) in dataset_field_descs:
                    dataset_field_descs[m.group(1)] = m.group(2).strip()
            PipelineModule._metadata_field_descriptions = deepcopy(
                (field_descs, dataset_field_descs)
            )

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

        # Output declarations belong to the class being documented.  Scanning
        # the whole module here leaks fields from sibling module classes that
        # happen to share the same file.  Sources from the effective processing
        # methods preserve inherited implementations without pulling in
        # overridden parent behavior; resolved metric-field attributes and
        # ``metric_info`` preserve dynamic setters.
        try:
            class_src = inspect.getsource(cls)
        except (TypeError, OSError):
            class_src = src
        output_sources = [class_src]
        for method_name in ("process", "process_batch", "post_process"):
            try:
                method_src = inspect.getsource(getattr(cls, method_name))
            except (AttributeError, TypeError, OSError):
                continue
            if method_src not in output_sources:
                output_sources.append(method_src)
        output_src = "\n".join(output_sources)

        # Input type: infer from process() checks
        needs_ref = "reference_path" in src
        needs_cap = bool(_re.search(r"caption.*\.text|\.caption", src[:3000]))
        video_only = bool(_re.search(r"not\s+sample\.is_video", src))
        audio_module = bool(
            _re.search(r"soundfile|librosa\.load|pesq|pystoi", src)
        ) or cls.name.startswith("audio_")
        batch_module = bool(
            _re.search(r"post_process.*all_samples|batch", src[:500]) and "def post_process" in src
        )

        if cls.input_type is not None:
            input_type = cls.input_type
        elif batch_module:
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
        for m in _re.finditer(r"quality_metrics\.(\w+)\s*=", output_src):
            field = m.group(1)
            if field not in outputs and field in field_descs:
                outputs[field] = field_descs[field]
        # Pattern 2: metric_field = "FIELD" / metric_field_name = "FIELD"
        # (base class or lightweight subclass auto-assignment)
        for m in _re.finditer(r'metric_field(?:_name)?\s*=\s*["\'](\w+)["\']', class_src):
            field = m.group(1)
            if field not in outputs and field in field_descs:
                outputs[field] = field_descs[field]
        for attr in ("metric_field", "metric_field_name"):
            field = getattr(cls, attr, None)
            if isinstance(field, str) and field in field_descs and field not in outputs:
                outputs[field] = field_descs[field]
        # Pattern 3: local aliases explicitly bound to ``quality_metrics``.
        aliases = set(_re.findall(r"\b(\w+)\s*=\s*(?:\w+\.)?quality_metrics\b", output_src))
        for alias in aliases:
            for m in _re.finditer(rf"\b{_re.escape(alias)}\.(\w+)\s*=", output_src):
                field = m.group(1)
                if field not in outputs and field in field_descs:
                    outputs[field] = field_descs[field]
            for m in _re.finditer(
                rf"setattr\(\s*{_re.escape(alias)}\s*,\s*[\"'](\w+)[\"']\s*,",
                output_src,
            ):
                field = m.group(1)
                if field not in outputs and field in field_descs:
                    outputs[field] = field_descs[field]
        # Explicit declarations cover dynamic/local aliases that the source
        # scanner cannot safely infer.  Only accept actual QualityMetrics
        # fields so dataset-level declarations remain separate below.
        for field, description in cls.metric_info.items():
            if field in field_descs and field not in outputs:
                outputs[field] = description or field_descs[field]

        dataset_outputs: Dict[str, str] = {}
        for m in _re.finditer(r'add_dataset_metric\(\s*["\'](\w+)["\']', output_src):
            field = m.group(1)
            if field not in dataset_outputs and field in dataset_field_descs:
                dataset_outputs[field] = dataset_field_descs[field]
        for field, description in cls.metric_info.items():
            if field in dataset_field_descs and field not in dataset_outputs:
                dataset_outputs[field] = description or dataset_field_descs[field]

        # Provenance metadata is always resolved per field (field -> class /
        # source / deviation), even when the module used a shorthand form.
        all_fields = list(outputs) + [f for f in dataset_outputs if f not in outputs]
        provenance_map = cls._resolve_field_map("provenance", all_fields)
        sources_map = cls._resolve_field_map("sources", all_fields)
        deviations_map = cls._resolve_field_map("deviations", all_fields)

        metadata = {
            "name": cls.name,
            "description": cls.description,
            "input_type": input_type,
            "output_fields": outputs,
            "dataset_output_fields": dataset_outputs,
            "default_config": dict(cls.default_config) if cls.default_config else {},
            "models": list(cls.models) if cls.models else [],
            "metric_info": dict(cls.metric_info) if cls.metric_info else {},
            "provenance": provenance_map,
            "sources": sources_map,
            "deviations": deviations_map,
            "requires_external_backend": bool(cls.requires_external_backend),
            "deprecated": bool(cls.deprecated),
        }
        cls._metadata_cache = metadata
        cls._metadata_cache_key = cache_key
        return deepcopy(metadata)

    @classmethod
    def _resolve_field_map(cls, attr: str, fields: Iterable[str]) -> Dict[str, str]:
        """Resolve a per-field class attribute to an explicit field->value map.

        A plain string applies to every declared field; a dict may carry a
        ``"*"`` default for unlisted fields. Entries for undeclared fields are
        kept too so typos stay visible to the provenance tests.
        """
        raw = getattr(cls, attr, None)
        if isinstance(raw, str):
            return {f: raw for f in fields}
        if not isinstance(raw, dict) or not raw:
            return {}
        default = raw.get("*")
        resolved: Dict[str, str] = {}
        for f in fields:
            value = raw.get(f, default)
            if value:
                resolved[f] = value
        for key, value in raw.items():
            if key != "*" and key not in resolved and value:
                resolved[key] = value
        return resolved

    @classmethod
    def field_provenance(cls) -> Dict[str, str]:
        """Field name -> provenance class for all declared output fields."""
        cached = cls.__dict__.get("_resolved_field_provenance")
        if cached is None:
            try:
                cached = dict(cls.get_metadata().get("provenance") or {})
            except Exception:
                cached = {}
            cls._resolved_field_provenance = cached
        return dict(cached)

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
    _RUNTIME_CONFIG_KEYS = {
        "parallel_jobs",
        "frame_cache_enabled",
        "timing_enabled",
        "cache_enabled",
    }

    @staticmethod
    def _deduplicate_modules(modules: List[PipelineModule]) -> List[PipelineModule]:
        """Collapse only duplicate canonical lip-sync compatibility requests."""
        result: List[PipelineModule] = []
        by_name: Dict[str, PipelineModule] = {}
        lip_sync_names = {"lip_sync_verse", "lip_sync_syncnet"}
        for module in modules:
            if module.name not in lip_sync_names:
                result.append(module)
                continue
            existing = by_name.get(module.name)
            if existing is None:
                by_name[module.name] = module
                result.append(module)
                continue
            if existing.config != module.config:
                differing = sorted(
                    key
                    for key in set(existing.config) | set(module.config)
                    if existing.config.get(key) != module.config.get(key)
                )
                raise ValueError(
                    f"Conflicting configurations for canonical module {module.name!r}; "
                    f"differing keys: {differing}"
                )
            for attr in (
                "_requested_module_name",
                "_legacy_output_aliases",
                "_legacy_lip_sync_protocol",
            ):
                incoming = getattr(module, attr, None)
                current = getattr(existing, attr, None)
                if incoming is not None and current is not None and incoming != current:
                    raise ValueError(
                        f"Conflicting compatibility modes for canonical module {module.name!r}"
                    )
                if incoming is not None and current is None:
                    setattr(existing, attr, deepcopy(incoming))
        return result

    def __init__(
        self,
        modules: List[PipelineModule],
        allow_provenance: Optional[Iterable[str]] = None,
    ):
        self.modules = self._deduplicate_modules(modules)
        self._legacy_module_aliases: Dict[str, str] = {}
        self._legacy_output_aliases: Dict[str, str] = {}
        self._legacy_lip_sync_protocol: Optional[str] = None
        for module in self.modules:
            requested_name = getattr(module, "_requested_module_name", None)
            if requested_name:
                existing = self._legacy_module_aliases.get(requested_name)
                if existing is not None and existing != module.name:
                    raise ValueError(
                        f"Legacy module {requested_name!r} resolves to both "
                        f"{existing!r} and {module.name!r}"
                    )
                self._legacy_module_aliases[requested_name] = module.name
            aliases = getattr(module, "_legacy_output_aliases", {})
            for canonical, legacy in aliases.items():
                if legacy in self._legacy_output_aliases.values():
                    raise ValueError(
                        f"Legacy output {legacy!r} is claimed by multiple lip-sync protocols"
                    )
                self._legacy_output_aliases[canonical] = legacy
            protocol = getattr(module, "_legacy_lip_sync_protocol", None)
            if protocol is not None:
                if (
                    self._legacy_lip_sync_protocol is not None
                    and self._legacy_lip_sync_protocol != protocol
                ):
                    raise ValueError("A pipeline cannot expose two legacy lip-sync protocols")
                self._legacy_lip_sync_protocol = protocol
        # Provenance gate: by default only "published" and "utility" fields let a
        # module run. Modules whose declared output fields are exclusively
        # "adapted"/"own" are excluded unless their classes were explicitly
        # allowed via the ``allow_provenance`` argument or the injected
        # ``[pipeline] allow_provenance`` config (a list like
        # ``["adapted", "own"]``). Modules declaring no provenance at all stay
        # allowed for backward compatibility (e.g. external plugins).
        valid_provenance = {"published", "adapted", "own", "utility"}
        self._allowed_provenance: Set[str] = {"published", "utility"}
        if isinstance(allow_provenance, str):
            allow_provenance = [allow_provenance]
        if allow_provenance:
            requested = {str(c).strip() for c in allow_provenance if str(c).strip()}
            unknown = requested - valid_provenance
            if unknown:
                raise ValueError(f"Unknown provenance classes: {sorted(unknown)}")
            self._allowed_provenance |= requested
        self._module_allowed_provenance: Dict[int, Set[str]] = {}
        for module in self.modules:
            module_allowed = set(self._allowed_provenance)
            extra = module.config.get("allow_provenance")
            if isinstance(extra, str):
                extra = [extra]
            if isinstance(extra, (list, tuple, set)):
                requested = {str(c).strip() for c in extra if str(c).strip()}
                unknown = requested - valid_provenance
                if unknown:
                    raise ValueError(
                        f"Unknown provenance classes for {module.name}: {sorted(unknown)}"
                    )
                module_allowed |= requested
            self._module_allowed_provenance[id(module)] = module_allowed
        self._provenance_excluded: Dict[str, str] = {}
        self._availability_excluded: Dict[str, str] = {}
        # field -> class map for stamping DatasetStats.metric_provenance from
        # add_dataset_metric() (the metric name is the only identifier there).
        self._dataset_field_provenance: Dict[str, str] = {}
        self._active_dataset_module: Optional[PipelineModule] = None
        for module in self.modules:
            try:
                meta = type(module).get_metadata()
            except Exception:
                continue
            declared = set(meta.get("output_fields", {})) | set(
                meta.get("dataset_output_fields", {})
            )
            prov_map = meta.get("provenance") or {}
            # Consider every marked output — including service side-channels
            # declared in ``provenance`` (e.g. ``detections`` on Sample) — so a
            # module with a utility role stays enabled even when its metrics
            # are own-class.
            classes = set(prov_map.values())
            if not classes and isinstance(getattr(type(module), "provenance", None), str):
                classes = {type(module).provenance}
            for f in meta.get("dataset_output_fields", {}):
                if f in prov_map and f not in self._dataset_field_provenance:
                    self._dataset_field_provenance[f] = prov_map[f]
            if classes and not (classes & self._module_allowed_provenance[id(module)]):
                self._provenance_excluded[module.name] = "+".join(sorted(classes))
        self.results: Dict[str, Sample] = {}
        self._result_signatures: Dict[str, tuple[object, ...]] = {}
        self._result_manifests: Dict[str, Dict[str, Any]] = {}
        self.stats = DatasetStats(total_samples=0, valid_samples=0, invalid_samples=0, total_size=0)
        self._batch_modules: List[PipelineModule] = []  # Modules that need batch processing
        self._hooks: Dict[str, Dict[str, Callable[[Sample], Sample]]] = {}
        self.module_timings: Dict[str, float] = {}
        self.module_call_counts: Dict[str, int] = {}
        self.module_failures: Dict[str, str] = {}
        # Per-sample frame store, keyed by file identity only. Each entry holds
        # one native-BGR decode plus lazily converted per-color variants; see
        # sample_frames() / load_representative_frame().
        self._frame_cache: Dict[tuple[Any, ...], Dict[str, Any]] = {}
        self._runtime_resource_cache: Dict[tuple[Any, ...], Any] = {}
        self._runtime_value_cache: Dict[tuple[Any, ...], Any] = {}
        # Result/frame/runtime caching is a GLOBAL feature gated by config
        # (``[general] cache_enabled``), which runtime_module_config() injects
        # identically onto every module. Semantics: the global config gates the
        # feature; a single module opting out (``cache_enabled: false``) no
        # longer disables caching for the whole run. Both result- and
        # frame-cache flags therefore aggregate with any() and are only off when
        # caching is disabled for every module (e.g. global config false, or a
        # benchmark profile that turned it off everywhere).
        # ``content_hash_keys: true`` (off by default) makes media cache keys
        # depend on a content digest rather than size+mtime.
        self._cache_enabled = (
            any(bool(module.config.get("cache_enabled", True)) for module in self.modules)
            if self.modules
            else True
        )
        self._content_hash_keys = (
            any(bool(module.config.get("content_hash_keys", False)) for module in self.modules)
            if self.modules
            else False
        )
        self._frame_cache_enabled = (
            any(bool(module.config.get("frame_cache_enabled", True)) for module in self.modules)
            if self.modules
            else True
        ) and self._cache_enabled
        self._timing_enabled = (
            any(bool(module.config.get("timing_enabled", True)) for module in self.modules)
            if self.modules
            else True
        )
        # Maps stats field name -> (QualityMetrics field name, count)
        self._AVG_METRIC_MAP: Dict[str, str] = {
            "avg_technical_score": "technical_score",
            "avg_aesthetic_score": "aesthetic_v25_score",
            "avg_motion_score": "motion_score",
        }
        self._metric_counts: Dict[str, int] = {k: 0 for k in self._AVG_METRIC_MAP}
        self._start_needs_reset = False

        # Give modules access to pipeline for batch metrics
        for module in self.modules:
            module.pipeline = self

    def _path_cache_state(self, path: Path) -> tuple[Any, ...]:
        """Return a file state tuple suitable for runtime cache keys.

        Base key is ``(resolved_path, size, mtime_ns)``; when content-hash
        keys are enabled, a content digest is appended.
        """
        resolved = str(Path(path).resolve())
        try:
            stat = Path(path).stat()
            base: tuple[Any, ...] = (resolved, stat.st_size, stat.st_mtime_ns)
        except OSError:
            base = (resolved, None, None)
        if self._content_hash_keys:
            from .runtime import content_digest

            return base + (content_digest(path),)
        return base

    def _frame_cache_key(self, kind: str, path: Path) -> tuple[Any, ...]:
        """Key the per-sample frame store by file identity only.

        Deliberately excludes ``max_frames`` and ``color`` so every module that
        touches the same file shares one decode. Different (max_frames, color)
        requests are served as subsampled views / lazily converted colors of
        that single decode.
        """
        return (kind, *self._path_cache_state(path))

    def sample_frames(self, path: Path, max_frames: int = 8, color: str = "rgb") -> List[Any]:
        """Load uniformly spaced frames, reusing a per-sample runtime cache.

        Returns a fresh list of zero-copy READ-ONLY numpy views each call. The
        exact ``max_frames`` sampling grid is decoded once in native BGR; color
        conversions for that grid are cached lazily. Keeping separate entries
        for different frame limits makes the result independent of request
        order. Callers must not mutate the returned arrays in place (copy first).
        """
        from .image import _convert_frame_color, _sample_frames_uncached

        max_frames = max(0, int(max_frames))
        color = str(color)

        if not self._frame_cache_enabled or max_frames <= 0:
            frames = _sample_frames_uncached(path, max_frames=max_frames, color=color)
            return cast(List[Any], clone_frames(frames))

        # A uniform N-frame grid generally is not a subset of a uniform M-frame
        # grid. Include max_frames so an earlier, denser request cannot change
        # the pixels returned by a later, smaller request.
        key = (*self._frame_cache_key("frames", Path(path)), max_frames)
        entry = self._frame_cache.get(key)
        if entry is None:
            decoded = _sample_frames_uncached(path, max_frames=max_frames, color="bgr")
            base = [readonly_view(frame) for frame in decoded]
            entry = {
                "bgr": base,
                "colors": {"bgr": base},
            }
            self._frame_cache[key] = entry

        full = entry["colors"].get(color)
        if full is None:
            full = [readonly_view(_convert_frame_color(frame, color)) for frame in entry["bgr"]]
            entry["colors"][color] = full

        return [readonly_view(frame) for frame in full]

    def load_representative_frame(self, path: Path, color: str = "rgb") -> Optional[Any]:
        """Load one representative frame, reusing a per-sample runtime cache.

        Returns a fresh zero-copy READ-ONLY numpy view (or ``None``). The middle
        frame is decoded once in native BGR and cached per file; each ``color``
        is converted-and-cached on first use. Do not mutate the result in place.
        """
        from .image import _load_representative_frame_uncached, _convert_frame_color

        color = str(color)
        if not self._frame_cache_enabled:
            frame = _load_representative_frame_uncached(path, color=color)
            return readonly_view(frame) if frame is not None else None

        key = self._frame_cache_key("representative", Path(path))
        entry = self._frame_cache.get(key)
        if entry is None:
            decoded = _load_representative_frame_uncached(path, color="bgr")
            base = readonly_view(decoded) if decoded is not None else None
            entry = {"bgr": base, "colors": ({} if base is None else {"bgr": base})}
            self._frame_cache[key] = entry

        base = entry["bgr"]
        if base is None:
            return None
        frame = entry["colors"].get(color)
        if frame is None:
            frame = readonly_view(_convert_frame_color(base, color))
            entry["colors"][color] = frame
        return readonly_view(frame)

    def _record_module_timing(self, module_name: str, elapsed: float, calls: int = 1) -> None:
        if not self._timing_enabled:
            return
        self.module_timings[module_name] = self.module_timings.get(module_name, 0.0) + elapsed
        self.module_call_counts[module_name] = self.module_call_counts.get(module_name, 0) + calls

    def get_timing_report(self) -> Dict[str, Dict[str, float]]:
        """Return per-module runtime totals for the latest pipeline run."""
        report: Dict[str, Dict[str, float]] = {}
        for name, total_seconds in self.module_timings.items():
            calls = self.module_call_counts.get(name, 0)
            report[name] = {
                "seconds": total_seconds,
                "calls": float(calls),
                "avg_seconds": total_seconds / calls if calls else 0.0,
            }
        return report

    def get_runtime_resource(self, key: tuple[Any, ...], factory: Callable[[], Any]) -> Any:
        """Return a shared resource for this pipeline run."""
        if not self._cache_enabled:
            return factory()
        if key not in self._runtime_resource_cache:
            self._runtime_resource_cache[key] = factory()
        return self._runtime_resource_cache[key]

    def get_runtime_value(self, key: tuple[Any, ...], factory: Callable[[], Any]) -> Any:
        """Return a shared value for the sample currently being processed."""
        if not self._cache_enabled:
            return factory()
        if key not in self._runtime_value_cache:
            self._runtime_value_cache[key] = factory()
        return self._runtime_value_cache[key]

    def peek_runtime_value(self, key: tuple[Any, ...], default: Any = None) -> Any:
        """Return a cached runtime value without constructing it."""
        return self._runtime_value_cache.get(key, default)

    def set_runtime_value(self, key: tuple[Any, ...], value: Any) -> None:
        """Store a value in the current runtime cache."""
        self._runtime_value_cache[key] = value

    @classmethod
    def _sample_cache_signature(cls, sample: Sample) -> tuple[object, ...]:
        """Return the parts of a sample that materially affect processing output."""
        caption = sample.caption
        input_context = {
            "video_metadata": (
                sample.video_metadata.model_dump(mode="json") if sample.video_metadata else None
            ),
            "image_metadata": (
                sample.image_metadata.model_dump(mode="json") if sample.image_metadata else None
            ),
            "audio_metadata": (
                sample.audio_metadata.model_dump(mode="json") if sample.audio_metadata else None
            ),
            "caption": caption.model_dump(mode="json") if caption else None,
            "detections": sample.detections,
            "embedding": sample.embedding,
            "metadata": sample.metadata,
        }
        normalized = cls._normalize_fingerprint_value(input_context)
        context_digest = hashlib.sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()
        return (
            str(sample.path),
            sample.is_video,
            str(sample.reference_path) if sample.reference_path else None,
            caption.text if caption else None,
            str(caption.source_file) if caption and caption.source_file else None,
            context_digest,
        )

    def _module_is_excluded(self, module_name: str) -> bool:
        return (
            module_name in self._provenance_excluded or module_name in self._availability_excluded
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
        try:
            import numpy as np

            if isinstance(value, np.generic):
                return cls._normalize_fingerprint_value(value.item())
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    return cls._normalize_fingerprint_value(value.tolist())
                contiguous = np.ascontiguousarray(value)
                return {
                    "__ndarray__": True,
                    "dtype": str(contiguous.dtype),
                    "shape": list(contiguous.shape),
                    "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
                }
        except ImportError:
            pass
        return str(value)

    @staticmethod
    def _module_source_digest(module: PipelineModule) -> str:
        """Return a stable digest of the module implementation used for scoring."""
        try:
            module_path = Path(inspect.getfile(module.__class__))
            source = module_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, TypeError):
            try:
                source = inspect.getsource(module.__class__)
            except (OSError, TypeError):
                source = f"{module.__class__.__module__}.{module.__class__.__qualname__}"
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def _runtime_environment_fingerprint(self) -> Dict[str, Any]:
        """Capture runtime versions that can materially change metric values."""
        import platform
        from importlib import metadata as importlib_metadata

        packages = {"numpy", "opencv-python", "pydantic", "torch", "transformers"}
        for module in self.modules:
            packages.update(str(package) for package in module.required_packages)

        versions: Dict[str, str] = {}
        for package in sorted(packages):
            try:
                versions[package] = importlib_metadata.version(package)
            except importlib_metadata.PackageNotFoundError:
                continue

        try:
            ayase_version = importlib_metadata.version("ayase")
        except importlib_metadata.PackageNotFoundError:
            ayase_version = str(getattr(sys.modules.get("ayase"), "__version__", "unknown"))

        return {
            "ayase": ayase_version,
            "python": platform.python_version(),
            "packages": versions,
        }

    def _pipeline_fingerprint(self) -> Dict[str, Any]:
        """Describe the active module stack for cache/state compatibility checks."""
        return {
            "runtime": self._runtime_environment_fingerprint(),
            "modules": [
                {
                    "name": module.name,
                    "class": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
                    "source_sha256": self._module_source_digest(module),
                    "config": self._normalize_fingerprint_value(module.config),
                    "models": self._normalize_fingerprint_value(module.models),
                    "test_mode": module.test_mode,
                }
                for module in self.modules
            ],
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
        return stat.st_size == snapshot.get("size") and stat.st_mtime_ns == snapshot.get("mtime_ns")

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

    @staticmethod
    def _issue_type_key(issue: "ValidationIssue") -> str:
        """Return a stable category key for ``issues_by_type`` aggregation.

        Prefers the structured ``issue_type`` field. Only when it is absent does
        it fall back to a message prefix, while rejecting Windows drive-letter
        prefixes such as ``"C"`` from ``"C:\\clip.mp4: too dark"``.
        """
        issue_type = getattr(issue, "issue_type", None)
        if issue_type:
            return str(issue_type)
        message = issue.message or ""
        head, sep, rest = message.partition(":")
        if (
            sep
            and head
            and len(head) <= 40
            and "\\" not in head
            and "/" not in head
            # Reject Windows drive letters, e.g. "C:\\..." / "D:/...".
            and not (len(head) == 1 and head.isalpha() and rest[:1] in ("\\", "/"))
        ):
            return head
        return message[:40] or "issue"

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
            dumped = qm.canonical_model_dump()
            for stats_field, qm_field in self._AVG_METRIC_MAP.items():
                # model_dump().get() silently yields None for removed fields
                # instead of emitting a DeprecationWarning per sample.
                self._update_average_stat(stats_field, dumped.get(qm_field), delta)

        for issue in sample.validation_issues:
            sev = issue.severity.value
            self._update_issue_counter(self.stats.severity_distribution, sev, delta)
            key = self._issue_type_key(issue)
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
        if sample.quality_metrics is not None and self._legacy_lip_sync_protocol is not None:
            sample.quality_metrics._set_lip_sync_legacy_protocol(self._legacy_lip_sync_protocol)
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
            canonical_name = self._legacy_module_aliases.get(module_name, module_name)
            self._hooks[canonical_name] = entry

    def add_dataset_metric(self, metric_name: str, value: Any) -> None:
        """Add a dataset-level metric to stats.

        Used by batch metric modules (FVD, KVD, etc.) to store their results.

        Args:
            metric_name: Name of the metric (e.g., "fvd", "kvd")
            value: Metric value
        """
        prov_class: Optional[str] = None
        active = self._active_dataset_module
        if active is not None:
            prov_class = type(active).field_provenance().get(metric_name)
            allowed = self._module_allowed_provenance.get(id(active), self._allowed_provenance)
            if prov_class and prov_class not in allowed:
                logger.warning(
                    "Ignoring dataset metric %s from %s: provenance %s is not allowed",
                    metric_name,
                    active.name,
                    prov_class,
                )
                return
        if prov_class is None:
            prov_class = self._dataset_field_provenance.get(metric_name)

        # Store in DatasetStats
        if hasattr(self.stats, metric_name):
            setattr(self.stats, metric_name, value)
            if prov_class:
                self.stats.metric_provenance[metric_name] = prov_class
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
        self.module_timings = {}
        self.module_call_counts = {}
        self.module_failures = {}
        self._availability_excluded = {}
        self._frame_cache = {}
        self._runtime_value_cache = {}
        if self._provenance_excluded:
            logger.info(
                "Provenance filter excluded %d module(s) (allowed: %s): %s. "
                "Set allow_provenance to run them.",
                len(self._provenance_excluded),
                ", ".join(sorted(self._allowed_provenance)),
                ", ".join(
                    f"{name} ({classes})"
                    for name, classes in sorted(self._provenance_excluded.items())
                ),
            )
        if self._availability_excluded:
            logger.warning(
                "External backend required for %d module(s): %s",
                len(self._availability_excluded),
                ", ".join(sorted(self._availability_excluded)),
            )
        for module in self.modules:
            if self._module_is_excluded(module.name):
                continue
            try:
                if not getattr(module, "_mounted", False):
                    module.on_mount()
                    # on_mount() sets _mounted = True only when setup() succeeds.
                    # Do NOT force-set it here — modules with missing packages
                    # must remain unmounted so process_sample() skips them.
            except Exception as e:
                logger.error(f"Error in on_mount for module {module.name}: {e}")
                self.module_failures[module.name] = f"mount failed: {type(e).__name__}: {e}"
            if not getattr(module, "_mounted", False):
                self.module_failures.setdefault(module.name, "module did not mount")
            elif (
                type(module).requires_external_backend
                and not getattr(module, "_backend", None)
                and not getattr(module, "_ml_available", False)
                and not getattr(module, "_available", False)
            ):
                self._availability_excluded[module.name] = "external_backend_unavailable"
            elif (
                getattr(module, "_backend", None) == "unavailable"
                and not getattr(module, "_ml_available", False)
                and not getattr(module, "_available", False)
            ):
                self.module_failures.setdefault(module.name, "backend unavailable")
        for module in self.modules:
            if module.name in self.module_failures or self._module_is_excluded(module.name):
                continue
            on_execute = getattr(module, "on_execute", None)
            if not callable(on_execute):
                continue
            try:
                on_execute()
            except Exception as e:
                logger.error(f"Error in on_execute for module {module.name}: {e}")
                self.module_failures[module.name] = f"on_execute failed: {type(e).__name__}: {e}"

    def get_run_status(self) -> Dict[str, Any]:
        """Return requested/mounted module coverage for the current run."""
        requested = [module.name for module in self.modules]
        failed = dict(self.module_failures)
        failed_samples = {
            path: list(sample.failed_modules)
            for path, sample in self.results.items()
            if sample.failed_modules
        }
        return {
            "complete": (
                not failed
                and not failed_samples
                and not self._availability_excluded
                and not self._provenance_excluded
            ),
            "requested_modules": requested,
            "mounted_modules": [
                name
                for name in requested
                if name not in failed and not self._module_is_excluded(name)
            ],
            "module_failures": failed,
            "failed_samples": failed_samples,
            "provenance_excluded": dict(self._provenance_excluded),
            "availability_excluded": dict(self._availability_excluded),
        }

    def _rebuild_sample_stats(self) -> None:
        """Recompute sample-derived aggregate stats from the stored results.

        Sample-derived fields (counts, averages, issue distributions) are kept
        in sync incrementally at ``_store_result`` time, but modules'
        ``post_process()`` may append issues (e.g. diversity/semantic selection)
        or flip validity *after* results were stored. Rebuilding here folds
        those late mutations into the final stats — and makes a resumed run
        (which rebuilds from restored samples) report the same numbers.
        Dataset-level and distribution fields not derived from samples are
        preserved.
        """
        previous = self.stats
        self.stats = DatasetStats(total_samples=0, valid_samples=0, invalid_samples=0, total_size=0)
        self._metric_counts = {k: 0 for k in self._AVG_METRIC_MAP}
        for field in DatasetStats.model_fields:
            if field in self._REBUILT_STATS_FIELDS:
                continue
            setattr(self.stats, field, getattr(previous, field))
        for sample in self.results.values():
            self._apply_sample_stats(sample, 1)

    def stop(self) -> None:
        """Finalize and cleanup all modules."""
        all_samples = list(self.results.values())

        # Call post_process on all modules first
        for module in self.modules:
            if module.name in self.module_failures or self._module_is_excluded(module.name):
                continue
            post_process = getattr(module, "post_process", None)
            if not callable(post_process):
                continue
            try:
                self._active_dataset_module = module
                post_process(all_samples)
            except Exception as e:
                logger.error(f"Error in post_process for module {module.name}: {e}")
                detail = f"post_process failed: {type(e).__name__}: {e}"
                self.module_failures[module.name] = detail
                for sample in all_samples:
                    self._register_module_failure(sample, module.name, detail)
            finally:
                self._active_dataset_module = None

        # post_process() can append issues / change validity after results were
        # stored; refold those into the sample-derived aggregate stats.
        self._rebuild_sample_stats()

        # Call on_dispose (this triggers batch metric computation)
        for module in self.modules:
            if self._module_is_excluded(module.name):
                continue
            on_dispose = getattr(module, "on_dispose", None)
            if not callable(on_dispose):
                continue
            try:
                self._active_dataset_module = module
                on_dispose()
            except Exception as e:
                logger.error(f"Error in on_dispose for module {module.name}: {e}")
                detail = f"on_dispose failed: {type(e).__name__}: {e}"
                self.module_failures[module.name] = detail
                for sample in all_samples:
                    self._register_module_failure(sample, module.name, detail)
            finally:
                self._active_dataset_module = None

        # on_dispose() (e.g. dedup batch modules) can flip validity or write
        # metrics onto samples AFTER post_process ran; refold those into the
        # sample-derived aggregate stats so exported numbers match the final
        # sample states. Rebuild recomputes from scratch, so it is idempotent
        # and does not double-count the earlier post_process rebuild.
        self._rebuild_sample_stats()

        # Log batch metrics if any were computed
        if self._batch_modules:
            logger.info(f"Processed {len(self._batch_modules)} batch metric modules")
            for module in self._batch_modules:
                logger.debug(f"  - {module.name}")
        self._runtime_resource_cache = {}
        self._start_needs_reset = True

    @staticmethod
    def _sample_is_complete(sample: Sample) -> bool:
        """A sample is complete only if no module failed while processing it.

        Incomplete samples are never served from cache and are reprocessed on
        the next run / resume, so a transient tooling failure does not get
        permanently baked in as a "successful" (but empty) result.
        """
        return not getattr(sample, "failed_modules", None)

    @staticmethod
    def _register_module_failure(sample: Sample, module_name: str, detail: str) -> None:
        """Mark a module failure on *sample* (visible + marks it incomplete).

        Records the module in ``sample.failed_modules`` (so the sample is
        reprocessed rather than cached) and appends a clearly-visible but
        NON-fatal ``WARNING`` issue: a tooling failure is not the same as bad
        data, so it must not flip ``is_valid`` to False.
        """
        if module_name not in sample.failed_modules:
            sample.failed_modules.append(module_name)
        message = f"Module '{module_name}' failed: {detail}"
        if not any(
            issue.issue_type == "module_error" and issue.message == message
            for issue in sample.validation_issues
        ):
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    issue_type="module_error",
                    message=message,
                    details={"module": module_name, "detail": detail},
                    recommendation=(
                        "Module raised or returned invalid output; this sample will "
                        "be reprocessed on the next run instead of served from cache."
                    ),
                )
            )

    @staticmethod
    def _restore_failure_state(
        sample: Sample,
        failed_modules: List[str],
        module_error_issues: List["ValidationIssue"],
    ) -> None:
        """Re-apply failure markers a fresh after-hook Sample would have dropped.

        An after-hook may return a brand-new ``Sample`` (e.g. ``model_copy``)
        that predates any ``failed_modules`` / ``module_error`` issue recorded
        while the module was processing (by the pipeline itself, or internally
        by a default ``process_batch``). Without this, a failed sample gets
        cached as complete and is never retried. Merges are idempotent (guarded
        by name / message) so re-running on an in-place-mutated sample is safe.
        """
        for name in failed_modules:
            if name not in sample.failed_modules:
                sample.failed_modules.append(name)
        if not module_error_issues:
            return
        existing = {
            issue.message
            for issue in sample.validation_issues
            if getattr(issue, "issue_type", None) == "module_error"
        }
        for issue in module_error_issues:
            if issue.message not in existing:
                sample.validation_issues.append(issue)

    @staticmethod
    def _snapshot_failure_state(
        sample: Sample,
    ) -> tuple[List[str], List["ValidationIssue"]]:
        """Capture failure markers so they can survive an after-hook revert."""
        failed = list(getattr(sample, "failed_modules", ()) or ())
        issues = [
            issue
            for issue in sample.validation_issues
            if getattr(issue, "issue_type", None) == "module_error"
        ]
        return failed, issues

    @staticmethod
    def _persist_backend(sample: Sample, module: "PipelineModule") -> None:
        """Persist a module's active backend/tier onto the sample's metrics.

        Zero per-module code required: any module that tracks ``self._backend``
        (a tier string such as "pyiqa"/"proxy"/"heuristic") has it recorded at
        ``quality_metrics.metric_backends[module.name]`` after a successful
        ``process()``. Modules with a falsy/absent ``_backend`` are skipped.
        """
        backend = getattr(module, "_backend", None)
        if not backend:
            return
        # A module that failed for this sample (e.g. default process_batch
        # caught its exception and returned the original sample) must not
        # advertise a backend as if it had produced metrics.
        if module.name in getattr(sample, "failed_modules", ()):
            return
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.metric_backends[module.name] = str(backend)

    @staticmethod
    def _metric_state(sample: Sample) -> Dict[str, Any]:
        """Snapshot metric values before a module runs."""
        import copy

        if sample.quality_metrics is None:
            return {}
        return copy.deepcopy(sample.quality_metrics.canonical_model_dump())

    @staticmethod
    def _metric_values_equal(left: Any, right: Any) -> bool:
        """Compare scalar or array-like metric values without ambiguous truth errors."""
        try:
            result = left == right
            if isinstance(result, bool):
                return result
            if hasattr(result, "all"):
                return bool(result.all())
            return bool(result)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _restore_metric_state(
        sample: Sample, module: "PipelineModule", before: Dict[str, Any]
    ) -> None:
        """Roll back declared metric fields after a failed module execution."""
        qm = sample.quality_metrics
        if qm is None:
            return
        provenance_before = before.get("metric_provenance", {})
        for field in type(module).field_provenance():
            if field not in type(qm).model_fields or field in qm._NON_METRIC_FIELDS:
                continue
            setattr(qm, field, deepcopy(before.get(field)))
            if isinstance(provenance_before, dict) and field in provenance_before:
                qm.metric_provenance[field] = provenance_before[field]
            else:
                qm.metric_provenance.pop(field, None)

    @classmethod
    def _reconcile_after_hook_metrics(
        cls,
        sample: Sample,
        module: "PipelineModule",
        before: Dict[str, Any],
        process_end: Dict[str, Any],
    ) -> None:
        """Remove module lineage from declared fields changed by an after hook."""
        qm = sample.quality_metrics
        if qm is None:
            return
        provenance_before = before.get("metric_provenance", {})
        provenance_process_end = process_end.get("metric_provenance", {})
        for field in type(module).field_provenance():
            if field not in type(qm).model_fields or field in qm._NON_METRIC_FIELDS:
                continue
            final = getattr(qm, field, None)
            if cls._metric_values_equal(process_end.get(field), final):
                if isinstance(provenance_process_end, dict) and field in provenance_process_end:
                    qm.metric_provenance[field] = provenance_process_end[field]
                else:
                    qm.metric_provenance.pop(field, None)
                continue
            if field in before and cls._metric_values_equal(before[field], final):
                if isinstance(provenance_before, dict) and field in provenance_before:
                    qm.metric_provenance[field] = provenance_before[field]
                else:
                    qm.metric_provenance.pop(field, None)
            else:
                qm.metric_provenance.pop(field, None)

    def _persist_provenance(
        self,
        sample: Sample,
        module: "PipelineModule",
        before: Optional[Dict[str, Any]] = None,
        written_fields: Optional[Set[str]] = None,
    ) -> None:
        """Stamp ``quality_metrics.metric_provenance`` for fields the module wrote.

        Zero per-module code required: after a successful ``process()``, every
        non-None output field declared by the module gets its provenance class
        recorded so each number in the output can be traced to published /
        adapted / own / utility. Skipped entirely for unmarked modules and for
        modules that failed on this sample.
        """
        if module.name in getattr(sample, "failed_modules", ()):
            return
        qm = sample.quality_metrics
        if qm is None:
            return
        try:
            prov = type(module).field_provenance()
        except Exception:
            return
        if not prov:
            return
        before = before or {}
        allowed = self._module_allowed_provenance.get(id(module), self._allowed_provenance)
        for field, cls_value in prov.items():
            if field not in type(qm).model_fields or field in qm._NON_METRIC_FIELDS:
                continue
            current = getattr(qm, field, None)
            was_written = written_fields is not None and field in written_fields
            if not was_written and (
                field in before and self._metric_values_equal(before[field], current)
            ):
                continue
            if not was_written and field not in before and current is None:
                continue
            if cls_value not in allowed:
                setattr(qm, field, deepcopy(before.get(field)))
                if field not in before or before.get(field) is None:
                    qm.metric_provenance.pop(field, None)
                continue
            if current is None:
                qm.metric_provenance.pop(field, None)
            else:
                qm.metric_provenance[field] = cls_value

    def process_sample(self, sample: Sample) -> Sample:
        """Run all active modules on a sample."""

        # Check if we already have a result for this file in memory (from load_state)
        str_path = str(sample.path)
        signature = self._sample_cache_signature(sample)
        manifest = self._sample_state_manifest(sample)
        cached = self.results.get(str_path)
        if (
            self._cache_enabled
            and cached is not None
            and self._result_signatures.get(str_path) == signature
            and self._result_manifests.get(str_path) == manifest
            and self._sample_is_complete(cached)
        ):
            return cached

        for module_name, detail in self.module_failures.items():
            self._register_module_failure(sample, module_name, detail)

        self._frame_cache = {}
        self._runtime_value_cache = {}
        try:
            with pipeline_context(self):
                for module in self.modules:
                    if (
                        not getattr(module, "_mounted", False)
                        or module.name in self.module_failures
                        or self._module_is_excluded(module.name)
                    ):
                        continue
                    started_at = perf_counter()
                    metric_state: Dict[str, Any] = {}
                    written_by_model: Dict[int, Set[str]] = {}
                    hooks = self._hooks.get(module.name)
                    entered = True
                    succeeded = False
                    try:
                        if hooks and "before" in hooks:
                            hook_raised = False
                            try:
                                hooked = hooks["before"](sample)
                            except Exception as e:
                                hook_raised = True
                                logger.error(
                                    "Before-hook for module %s raised for %s: %s; "
                                    "skipping module",
                                    module.name,
                                    str_path,
                                    e,
                                )
                                hooked = None
                                self._register_module_failure(
                                    sample,
                                    module.name,
                                    f"before hook raised {type(e).__name__}: {e}",
                                )
                            if not isinstance(hooked, Sample):
                                if hooked is not None:
                                    logger.error(
                                        "Before-hook for module %s returned %s for %s; "
                                        "skipping module",
                                        module.name,
                                        type(hooked).__name__,
                                        str_path,
                                    )
                                    self._register_module_failure(
                                        sample,
                                        module.name,
                                        f"before hook returned {type(hooked).__name__}, "
                                        "expected Sample",
                                    )
                                elif not hook_raised:
                                    self._register_module_failure(
                                        sample,
                                        module.name,
                                        "before hook returned None, expected Sample",
                                    )
                                entered = False
                                continue
                            sample = hooked

                        metric_state = self._metric_state(sample)

                        def record_write(metrics: QualityMetrics, field: str) -> None:
                            written_by_model.setdefault(id(metrics), set()).add(field)

                        try:
                            with observe_metric_writes(record_write):
                                processed = module.process(sample)
                        except Exception as e:
                            logger.error(
                                f"Error in module {module.name} for "
                                f"{getattr(sample, 'path', str_path)}: {e}"
                            )
                            self._register_module_failure(
                                sample, module.name, f"{type(e).__name__}: {e}"
                            )
                            self._restore_metric_state(sample, module, metric_state)
                        else:
                            if isinstance(processed, Sample):
                                sample = processed
                                succeeded = True
                            else:
                                logger.error(
                                    "Module %s returned %s for %s; keeping previous sample",
                                    module.name,
                                    type(processed).__name__,
                                    str_path,
                                )
                                self._register_module_failure(
                                    sample,
                                    module.name,
                                    f"returned {type(processed).__name__}, expected Sample",
                                )
                                self._restore_metric_state(sample, module, metric_state)
                    finally:
                        process_end_state: Dict[str, Any] = {}
                        if succeeded:
                            self._persist_backend(sample, module)
                            qm = sample.quality_metrics
                            self._persist_provenance(
                                sample,
                                module,
                                metric_state,
                                written_by_model.get(id(qm)) if qm is not None else None,
                            )
                            process_end_state = self._metric_state(sample)
                        # Always revert before-hook state so mutations do not
                        # propagate to later modules or the stored result.
                        if entered and hooks and "after" in hooks:
                            # Snapshot failure markers before the revert; an
                            # after-hook that returns a fresh Sample would
                            # otherwise discard them (caching a failed sample as
                            # complete so it is never retried).
                            pre_failed, pre_issues = self._snapshot_failure_state(sample)
                            restored: Optional[Sample]
                            try:
                                restored = hooks["after"](sample)
                            except Exception as e:
                                logger.error(
                                    "After-hook for module %s raised for %s: %s; "
                                    "keeping module output",
                                    module.name,
                                    str_path,
                                    e,
                                )
                                restored = None
                                self._register_module_failure(
                                    sample,
                                    module.name,
                                    f"after hook raised {type(e).__name__}: {e}",
                                )
                            if isinstance(restored, Sample):
                                sample = restored
                                self._restore_failure_state(sample, pre_failed, pre_issues)
                            elif restored is not None:
                                logger.error(
                                    "After-hook for module %s returned %s for %s; "
                                    "keeping module output",
                                    module.name,
                                    type(restored).__name__,
                                    str_path,
                                )
                                self._register_module_failure(
                                    sample,
                                    module.name,
                                    f"after hook returned {type(restored).__name__}, "
                                    "expected Sample",
                                )
                        if succeeded and process_end_state:
                            self._reconcile_after_hook_metrics(
                                sample, module, metric_state, process_end_state
                            )
                        self._record_module_timing(module.name, perf_counter() - started_at)
        finally:
            self._frame_cache = {}
            self._runtime_value_cache = {}

        # Cache the result and keep aggregate stats in sync.
        self._store_result(str_path, sample, signature=signature, manifest=manifest)

        return sample

    @staticmethod
    def _coerce_batch_size(batch_size: Optional[int]) -> int:
        try:
            return max(1, int(batch_size or 1))
        except (TypeError, ValueError):
            return 1

    def _process_module_batch(
        self,
        module: PipelineModule,
        samples: List[Sample],
    ) -> List[Sample]:
        """Run one module across a sample batch while preserving hook semantics."""

        if not samples:
            return []

        started_at = perf_counter()
        working = list(samples)
        hooks = self._hooks.get(module.name)
        eligible_samples: List[Sample] = []
        eligible_positions: List[int] = []
        succeeded_positions: List[int] = []
        metric_states: Dict[int, Dict[str, Any]] = {}
        written_by_model: Dict[int, Set[str]] = {}
        try:
            for idx, sample in enumerate(working):
                str_path = str(sample.path)
                if hooks and "before" in hooks:
                    hook_raised = False
                    try:
                        hooked = hooks["before"](sample)
                    except Exception as e:
                        hook_raised = True
                        logger.error(
                            "Before-hook for module %s raised for %s: %s; " "skipping module",
                            module.name,
                            str_path,
                            e,
                        )
                        hooked = None
                        self._register_module_failure(
                            working[idx],
                            module.name,
                            f"before hook raised {type(e).__name__}: {e}",
                        )
                    if not isinstance(hooked, Sample):
                        if hooked is not None:
                            logger.error(
                                "Before-hook for module %s returned %s for %s; " "skipping module",
                                module.name,
                                type(hooked).__name__,
                                str_path,
                            )
                            self._register_module_failure(
                                working[idx],
                                module.name,
                                f"before hook returned {type(hooked).__name__}, " "expected Sample",
                            )
                        elif not hook_raised:
                            self._register_module_failure(
                                working[idx],
                                module.name,
                                "before hook returned None, expected Sample",
                            )
                        continue
                    working[idx] = hooked
                    sample = hooked

                eligible_samples.append(sample)
                eligible_positions.append(idx)
                metric_states[idx] = self._metric_state(sample)

            if eligible_samples:
                try:

                    def record_write(metrics: QualityMetrics, field: str) -> None:
                        written_by_model.setdefault(id(metrics), set()).add(field)

                    with observe_metric_writes(record_write):
                        processed_batch = module.process_batch(eligible_samples)
                except Exception as e:
                    sample_path = getattr(eligible_samples[0], "path", "<batch>")
                    logger.error(
                        f"Error in module {module.name} for batch starting at "
                        f"{sample_path}: {e}"
                    )
                    for pos in eligible_positions:
                        self._register_module_failure(
                            working[pos], module.name, f"{type(e).__name__}: {e}"
                        )
                        self._restore_metric_state(working[pos], module, metric_states.get(pos, {}))
                else:
                    if not isinstance(processed_batch, list):
                        logger.error(
                            "Module %s returned %s for a batch; keeping previous samples",
                            module.name,
                            type(processed_batch).__name__,
                        )
                        for pos in eligible_positions:
                            self._register_module_failure(
                                working[pos],
                                module.name,
                                f"process_batch returned {type(processed_batch).__name__}, "
                                "expected list",
                            )
                            self._restore_metric_state(
                                working[pos], module, metric_states.get(pos, {})
                            )
                    elif len(processed_batch) != len(eligible_samples):
                        logger.error(
                            "Module %s returned %d samples for a batch of %d; "
                            "keeping previous samples",
                            module.name,
                            len(processed_batch),
                            len(eligible_samples),
                        )
                        for pos in eligible_positions:
                            self._register_module_failure(
                                working[pos],
                                module.name,
                                f"process_batch returned {len(processed_batch)} samples "
                                f"for a batch of {len(eligible_samples)}",
                            )
                            self._restore_metric_state(
                                working[pos], module, metric_states.get(pos, {})
                            )
                    else:
                        for pos, previous, processed in zip(
                            eligible_positions,
                            eligible_samples,
                            processed_batch,
                        ):
                            if not isinstance(processed, Sample):
                                logger.error(
                                    "Module %s returned %s for %s; keeping previous sample",
                                    module.name,
                                    type(processed).__name__,
                                    previous.path,
                                )
                                self._register_module_failure(
                                    working[pos],
                                    module.name,
                                    f"returned {type(processed).__name__}, expected Sample",
                                )
                                self._restore_metric_state(
                                    working[pos], module, metric_states.get(pos, {})
                                )
                                continue
                            working[pos] = processed
                            succeeded_positions.append(pos)
        finally:
            # Always revert hook state for entered positions (consistent with
            # the single-sample path), then persist backends for the positions
            # that produced valid output, then record timing.
            process_end_states: Dict[int, Dict[str, Any]] = {}
            for pos in succeeded_positions:
                self._persist_backend(working[pos], module)
                qm = working[pos].quality_metrics
                self._persist_provenance(
                    working[pos],
                    module,
                    metric_states.get(pos),
                    written_by_model.get(id(qm)) if qm is not None else None,
                )
                process_end_states[pos] = self._metric_state(working[pos])
            if hooks and "after" in hooks:
                for pos in eligible_positions:
                    # Snapshot failure markers before the revert. An after-hook
                    # returning a fresh Sample (or a marker recorded internally
                    # by the default process_batch) would otherwise be dropped,
                    # caching a failed sample as complete so it is never retried.
                    pre_failed, pre_issues = self._snapshot_failure_state(working[pos])
                    restored: Optional[Sample]
                    try:
                        restored = hooks["after"](working[pos])
                    except Exception as e:
                        logger.error(
                            "After-hook for module %s raised for %s: %s; " "keeping module output",
                            module.name,
                            working[pos].path,
                            e,
                        )
                        restored = None
                        self._register_module_failure(
                            working[pos],
                            module.name,
                            f"after hook raised {type(e).__name__}: {e}",
                        )
                    if isinstance(restored, Sample):
                        working[pos] = restored
                        self._restore_failure_state(working[pos], pre_failed, pre_issues)
                    elif restored is not None:
                        logger.error(
                            "After-hook for module %s returned %s for %s; " "keeping module output",
                            module.name,
                            type(restored).__name__,
                            working[pos].path,
                        )
                        self._register_module_failure(
                            working[pos],
                            module.name,
                            f"after hook returned {type(restored).__name__}, expected Sample",
                        )
                    if pos in process_end_states:
                        self._reconcile_after_hook_metrics(
                            working[pos],
                            module,
                            metric_states.get(pos, {}),
                            process_end_states[pos],
                        )
            self._record_module_timing(
                module.name,
                perf_counter() - started_at,
                calls=len(samples),
            )

        return working

    def _process_sample_batch(self, samples: List[Sample]) -> List[Sample]:
        """Run all active modules over a batch and store results in input order."""

        if not samples:
            return []

        outputs: List[Optional[Sample]] = [None] * len(samples)
        pending_samples: List[Sample] = []
        pending_meta: List[tuple[int, str, tuple[object, ...], Dict[str, Any]]] = []

        for idx, sample in enumerate(samples):
            str_path = str(sample.path)
            signature = self._sample_cache_signature(sample)
            manifest = self._sample_state_manifest(sample)
            cached = self.results.get(str_path)
            if (
                self._cache_enabled
                and cached is not None
                and self._result_signatures.get(str_path) == signature
                and self._result_manifests.get(str_path) == manifest
                and self._sample_is_complete(cached)
            ):
                outputs[idx] = cached
                continue

            pending_samples.append(sample)
            pending_meta.append((idx, str_path, signature, manifest))

        for sample in pending_samples:
            for module_name, detail in self.module_failures.items():
                self._register_module_failure(sample, module_name, detail)

        if pending_samples:
            active_samples = pending_samples
            self._frame_cache = {}
            self._runtime_value_cache = {}
            try:
                with pipeline_context(self):
                    for module in self.modules:
                        if (
                            not getattr(module, "_mounted", False)
                            or module.name in self.module_failures
                        ):
                            continue
                        active_samples = self._process_module_batch(module, active_samples)
            finally:
                self._frame_cache = {}
                self._runtime_value_cache = {}

            for (idx, str_path, signature, manifest), sample in zip(pending_meta, active_samples):
                self._store_result(str_path, sample, signature=signature, manifest=manifest)
                outputs[idx] = sample

        return [sample for sample in outputs if sample is not None]

    def process_samples(
        self,
        samples: Iterable[Sample],
        *,
        batch_size: Optional[int] = None,
    ) -> List[Sample]:
        """Run all active modules on multiple samples.

        ``batch_size=1`` preserves the legacy per-sample execution path. Larger
        values process samples module-by-module within each chunk, allowing
        modules to override ``process_batch()`` for true batched inference.
        """

        size = self._coerce_batch_size(batch_size)
        if size <= 1:
            return [self.process_sample(sample) for sample in samples]

        processed: List[Sample] = []
        batch: List[Sample] = []
        for sample in samples:
            batch.append(sample)
            if len(batch) >= size:
                processed.extend(self._process_sample_batch(batch))
                batch = []
        if batch:
            processed.extend(self._process_sample_batch(batch))
        return processed

    def export_report(self, path: Path, format: str = "json") -> None:
        """Export a detailed validation report.

        Args:
            path: Output file path
            format: 'json', 'csv', or 'html'
        """
        if format not in {"json", "csv", "html"}:
            raise ValueError(f"Unsupported report format: {format!r}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if format == "json":
            with open(path, "w", encoding="utf-8") as f:
                data = {
                    "run_status": self.get_run_status(),
                    "stats": self.stats.model_dump(),
                    "samples": [self.dump_sample(s) for s in self.results.values()],
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
                    score = (
                        s.quality_metrics.canonical_model_dump().get("technical_score")
                        if s.quality_metrics
                        else None
                    ) or 0.0

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

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Recursively coerce *value* into JSON-native types.

        Handles the values a module may stash in ``Sample.metadata``
        (``Dict[str, Any]``): numpy scalars/arrays, ``Path``, sets, and any
        other object, none of which pydantic's JSON serializer accepts.
        """
        if isinstance(value, dict):
            return {str(k): Pipeline._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Pipeline._json_safe(v) for v in value]
        if isinstance(value, (set, frozenset)):
            return [Pipeline._json_safe(v) for v in value]
        if isinstance(value, (str, bool, int, float)) or value is None:
            return value
        try:
            import numpy as _np

            if isinstance(value, _np.generic):
                return Pipeline._json_safe(value.item())
            if isinstance(value, _np.ndarray):
                return Pipeline._json_safe(value.tolist())
        except Exception:
            pass
        if isinstance(value, Path):
            return str(value)
        return str(value)

    def dump_sample(self, sample: Sample) -> Dict[str, Any]:
        """Serialize a public result, applying requested legacy output names once."""
        dumped = self._dump_sample_state(str(sample.path), sample)
        if not self._legacy_output_aliases:
            return dumped
        metrics = dumped.get("quality_metrics")
        if not isinstance(metrics, dict):
            return dumped
        dumped["quality_metrics"] = move_lip_sync_legacy_aliases(
            metrics, self._legacy_lip_sync_protocol
        )
        return dumped

    @classmethod
    def _dump_sample_state(cls, key: str, sample: Sample) -> Dict[str, Any]:
        """Serialize a sample to JSON-safe dict, tolerating bad ``metadata``.

        A module may store a non-JSON value (numpy scalar/array, ``Path``, ...)
        in ``Sample.metadata``, which makes ``model_dump(mode="json")`` raise.
        A single such sample must not abort the whole ``save_state`` (which
        would drop the entire run's resume cache). On failure, retry with
        ``metadata`` coerced to JSON-safe types; keep valid metadata intact.
        """
        try:
            return cast(Dict[str, Any], sample.canonical_model_dump(mode="json"))
        except Exception as exc:
            logger.warning(
                "Sample %s has non-serializable metadata; coercing to JSON-safe "
                "types for state save: %s",
                key,
                exc,
            )
        try:
            safe = sample.model_copy(update={"metadata": cls._json_safe(sample.metadata)})
            return cast(Dict[str, Any], safe.canonical_model_dump(mode="json"))
        except Exception as exc:
            logger.warning(
                "Sample %s metadata still non-serializable after coercion; "
                "dropping metadata for state save: %s",
                key,
                exc,
            )
            dropped = sample.model_copy(update={"metadata": {}})
            return cast(Dict[str, Any], dropped.canonical_model_dump(mode="json"))

    def save_state(self, path: Path) -> None:
        """Save current pipeline state to disk for resume."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "pipeline_fingerprint": self._pipeline_fingerprint(),
                "results": {k: self._dump_sample_state(k, v) for k, v in self.results.items()},
                "stats": self.stats.model_dump(mode="json"),
                "cache_manifest": {
                    k: self._result_manifests.get(k) or self._sample_state_manifest(v)
                    for k, v in self.results.items()
                },
            }
            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    # default=str is a final backstop for any stray non-JSON
                    # value elsewhere so one bad value can't sink the save.
                    json.dump(data, f, indent=2, default=str)
                Path(tmp_path).replace(path)
            except Exception:
                Path(tmp_path).unlink(missing_ok=True)
                raise
            logger.info(f"State saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    @staticmethod
    def _sanitize_cached_sample(key: str, data: Any) -> Any:
        """Drop metric keys unknown to the current QualityMetrics schema.

        Older state files may reference metric fields that have since been
        removed. QualityMetrics forbids extra keys, so strip them (with a
        warning) rather than failing to restore the whole cached sample.
        """
        if not isinstance(data, dict):
            return data
        qm = data.get("quality_metrics")
        if not isinstance(qm, dict):
            return data

        from .models import QualityMetrics

        legacy_metric_keys = {"lse_c", "lse_d"}
        unknown = [
            k for k in qm if k not in QualityMetrics.model_fields and k not in legacy_metric_keys
        ]
        if not unknown:
            return data
        logger.warning(
            "Cached sample %s has %d unknown metric field(s) from an older "
            "version; dropping: %s",
            key,
            len(unknown),
            ", ".join(sorted(unknown)),
        )
        cleaned = {k: val for k, val in qm.items() if k not in unknown}
        return {**data, "quality_metrics": cleaned}

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
                        context = (
                            {"lip_sync_protocol": self._legacy_lip_sync_protocol}
                            if self._legacy_lip_sync_protocol is not None
                            else None
                        )
                        sample = Sample.model_validate(
                            self._sanitize_cached_sample(k, v), context=context
                        )
                        if (
                            sample.quality_metrics is not None
                            and self._legacy_lip_sync_protocol is not None
                        ):
                            sample.quality_metrics.method_set_lip_sync_legacy_protocol(
                                self._legacy_lip_sync_protocol
                            )
                        manifest = manifests.get(k) if isinstance(manifests, dict) else None
                        if isinstance(manifest, dict):
                            if not self._sample_matches_manifest(manifest):
                                logger.info(
                                    f"Skipping stale cache for {k} (state manifest changed)"
                                )
                                partial_restore = True
                                continue
                        elif not self._sample_matches_basic_cache(sample):
                            logger.info(f"Skipping stale cache for {k} (file changed or missing)")
                            partial_restore = True
                            continue
                        self.results[k] = sample
                        self._result_signatures[k] = self._sample_cache_signature(sample)
                        self._result_manifests[k] = (
                            manifest
                            if isinstance(manifest, dict)
                            else self._sample_state_manifest(sample)
                        )
                        self._apply_sample_stats(sample, 1)
                    except Exception as e:
                        partial_restore = True
                        logger.warning(f"Failed to load sample {k}: {e}")

            if (
                saved_stats is not None
                and not partial_restore
                and len(self.results) == len(results_data)
            ):
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


# Renamed module registry names kept resolvable for consumers that still
# reference the pre-0.1.80 identifiers. Aliases resolve in ``get_module()``
# only — they are not listed by ``list_modules()`` and emit a
# DeprecationWarning on use.
MODULE_ALIASES: Dict[str, str] = {
    "clifvqa": "clip_feel",
    "entitybench": "entity_consistency",
    "geneval": "clip_prompt_check",
    "graphsim": "local_std_uniformity",
    "jedi": "mmd_selfsplit",
    "jedi_metric": "mmd_selfsplit_metric",
    "magface": "face_emb_norm",
    "modularbvqa": "clip_slowfast_vq",
    "movie": "gabor_flow_vq",
    "multiple_objects": "object_count_check",
    "naturalness": "brisque_inverted",
    "pcqm": "mse_color",
    "pointssim": "chamfer_sim",
    "psnr_div": "psnr_grad",
    "psnr_hvs": "psnr_hvs_approx",
    "pu_metrics": "log_metrics",
    "spherical_psnr": "erp_psnr",
    "st_greed": "mscn_entropy",
    "st_lpips": "stlpips_selfdist",
    "t2v_score": "t2v_generic_score",
    "tc_bench": "clip_event_order",
    "tlvqm": "resnet_svr_vq",
    "ttsds2": "tts_system_dist",
    "vader": "hpsv2_const",
    "videophy": "vlm_phy",
    "videval": "svr60_vq",
    "audio_lpdist": "audio_logmel_dist",
    "finevq": "finevq_raw",
    "hdr_vqm": "hdr_subband_flicker_score",
    "dynamics_range": "content_variation",
    "nima_onnx": "nima",
}


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
        module_cls = cls._modules.get(name)
        if module_cls is None and name == "lip_sync":
            # ``lip_sync`` is a config-aware compatibility factory and is
            # intentionally absent from the canonical module registry.
            try:
                from .modules.lip_sync import LipSyncModule

                module_cls = LipSyncModule
            except ImportError:
                module_cls = None
        if module_cls is None:
            new_name = MODULE_ALIASES.get(name)
            if new_name is not None and new_name in cls._modules:
                warnings.warn(
                    f"Module name '{name}' was renamed to '{new_name}' in 0.1.80; "
                    "the alias will be removed in a later release.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                module_cls = cls._modules[new_name]
        return module_cls

    @classmethod
    def is_packaged_module(cls, module_cls: Type[PipelineModule]) -> bool:
        """Return whether a registered module ships from ``ayase.modules``."""
        return getattr(module_cls, "__module__", "").startswith("ayase.modules.")

    @classmethod
    def requires_external_backend(cls, name: str) -> bool:
        """Return whether a registered module is flagged ``requires_external_backend``.

        External-backend modules have no turnkey real backend in a standard install;
        they stay registered/revivable but are excluded from the documented
        module/metric counts and listed in an External backend required section instead.
        Unknown names return False.
        """
        module_cls = cls._modules.get(name)
        return bool(
            module_cls is not None and getattr(module_cls, "requires_external_backend", False)
        )

    @classmethod
    def list_modules(
        cls, packaged_only: bool = False, include_external_backends: bool = True
    ) -> Dict[str, str]:
        """Return dict of name -> description, sorted by name for stable iteration.

        ``include_external_backends=False`` drops modules flagged ``requires_external_backend`` so
        doc generators can report only the delivered (real-backend) set.
        """
        return {
            name: cls._modules[name].description
            for name in sorted(cls._modules)
            if (not packaged_only or cls.is_packaged_module(cls._modules[name]))
            and (
                include_external_backends
                or not getattr(cls._modules[name], "requires_external_backend", False)
            )
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
        cls._register_metric_groups()

    @classmethod
    def _register_metric_groups(cls) -> None:
        """Fold each module's ``metric_groups`` into QualityMetrics' registry."""
        from .models import QualityMetrics

        for module_cls in cls._modules.values():
            groups = getattr(module_cls, "metric_groups", None)
            if groups:
                QualityMetrics.register_field_groups(groups)

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
                resolved_folder = folder.resolve()
                for file_path in folder.glob("*.py"):
                    if file_path.name.startswith("_"):
                        continue
                    readiness_label = cls._external_plugin_label(file_path)
                    resolved_file = file_path.resolve()
                    # Plugin folders are trusted-code roots. Do not let a file
                    # symlink turn discovery into execution of Python outside
                    # the explicitly configured root.
                    if resolved_file.parent != resolved_folder:
                        cls._record_readiness(
                            readiness_label,
                            False,
                            "Plugin path resolves outside configured folder",
                        )
                        current_labels.add(readiness_label)
                        logger.warning(
                            "Skipping external plugin outside configured folder: %s",
                            file_path,
                        )
                        continue
                    file_key = str(resolved_file)
                    current_file_keys.add(file_key)
                    current_labels.add(readiness_label)
                    previous_module_name = cls._external_plugin_modules.get(file_key)
                    if previous_module_name is not None:
                        sys.modules.pop(previous_module_name, None)
                        cls._remove_registered_classes_for_module(previous_module_name)
                    # Stable digest of the normalized path: builtin hash() is
                    # per-process randomized, so it would make the plugin's
                    # module name (and thus the pipeline fingerprint) differ
                    # every run, silently invalidating resume/state matching.
                    file_digest = hashlib.sha1(file_key.encode("utf-8")).hexdigest()[:10]
                    module_name = f"ayase_ext_{file_path.stem}_{file_digest}"
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


def instantiate_module_requests(
    requests: Iterable[tuple[str, Dict[str, Any]]],
) -> List[PipelineModule]:
    """Resolve module requests, then deduplicate their canonical instances."""
    modules: List[PipelineModule] = []
    for requested_name, params in requests:
        module_cls = ModuleRegistry.get_module(requested_name)
        if module_cls is None:
            raise ValueError(f"Unknown module: {requested_name}")
        module = module_cls(config=deepcopy(params))
        if requested_name == "lip_sync" and not getattr(module, "_requested_module_name", None):
            setattr(module, "_requested_module_name", "lip_sync")
        modules.append(module)
    return Pipeline._deduplicate_modules(modules)


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
        clones: List[PipelineModule] = []
        for module in source:
            clone = module.__class__(config=deepcopy(module.config))
            for attr in (
                "_requested_module_name",
                "_legacy_output_aliases",
                "_legacy_lip_sync_protocol",
            ):
                if hasattr(module, attr):
                    setattr(clone, attr, deepcopy(getattr(module, attr)))
            clones.append(clone)
        return clones

    def _build_modules(self, names: List[str]) -> List[PipelineModule]:
        # Modules named via ``modules=`` or ``pipeline.modules`` are an explicit
        # choice, which is the provenance opt-in: adapted/own modules may run.
        requests: List[tuple[str, Dict[str, Any]]] = []
        for name in names:
            requests.append((name, opt_in_all_provenance(runtime_module_config(self.config))))
        return instantiate_module_requests(requests)

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
            batch_size = getattr(self.config.general, "sample_batch_size", 1)
            pipeline.process_samples(samples, batch_size=batch_size)
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

"""Registration and discovery of packaged metric modules and external plugins."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import pkgutil
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type

from .models import QualityMetrics

if TYPE_CHECKING:
    from .pipeline import PipelineModule

logger = logging.getLogger("ayase.pipeline")


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
        if existing is None and module_cls.__module__.startswith("ayase_ext_"):
            existing = cls._load_indexed(module_cls.name)
        if existing is not None and existing is not module_cls:
            raise ValueError(
                f"Duplicate module name '{module_cls.name}' for "
                f"{existing.__module__}.{existing.__name__} and "
                f"{module_cls.__module__}.{module_cls.__name__}"
            )
        cls._modules[module_cls.name] = module_cls

    @classmethod
    def get_module(cls, name: str) -> Optional[Type[PipelineModule]]:
        module_cls = cls._modules.get(name) or cls._load_indexed(name)
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
            if new_name is not None:
                module_cls = cls._modules.get(new_name) or cls._load_indexed(new_name)
            if module_cls is not None:
                warnings.warn(
                    f"Module name '{name}' was renamed to '{new_name}' in 0.1.80; "
                    "the alias will be removed in a later release.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        return module_cls

    @classmethod
    def _load_indexed(cls, name: str) -> Optional[Type[PipelineModule]]:
        from .pipeline import load_module_index

        entry = load_module_index().get(name)
        if entry is None:
            return None
        module_name = entry["import_path"]
        was_imported = module_name in sys.modules
        previous_modules = dict(cls._modules)
        try:
            imported: Any = importlib.import_module(module_name)
            for component in entry["class_name"].split("."):
                imported = getattr(imported, component)
            registered = cls._modules.get(name)
            if registered is None or registered is not imported:
                raise ImportError(f"Module {module_name} did not register {name!r} as indexed")
        except Exception as exc:
            if not was_imported:
                sys.modules.pop(module_name, None)
            cls._rollback_partial_registration(previous_modules, module_name)
            cls._record_readiness(module_name.rsplit(".", 1)[-1], False, str(exc))
            logger.warning("Failed to import module %s: %s", module_name, exc)
            return None
        cls._record_readiness(module_name.rsplit(".", 1)[-1], True)
        QualityMetrics.register_field_groups(entry["metric_groups"])
        return registered

    @classmethod
    def prepare_named_lookup(cls, plugin_paths: List[Path]) -> None:
        """Load configured plugins without importing the entire built-in catalog."""
        from .pipeline import load_module_index

        index = load_module_index()
        if not index:
            cls.discover_modules(plugin_paths=plugin_paths)
            return
        for entry in index.values():
            QualityMetrics.register_field_groups(entry["metric_groups"])
        if plugin_paths:
            cls.discover_external_modules(plugin_paths)
        cls._register_metric_groups()

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
    def is_unimplemented(cls, name: str) -> bool:
        """Return whether a registered module has unresolved implementation work."""
        module_cls = cls._modules.get(name)
        return bool(
            module_cls is not None
            and module_cls.resolved_implementation_status() == "unimplemented"
        )

    @classmethod
    def list_modules(
        cls, packaged_only: bool = False, include_external_backends: bool = True
    ) -> Dict[str, str]:
        """Return dict of name -> description, sorted by name for stable iteration.

        ``include_external_backends=False`` drops unimplemented modules so doc
        generators can report only the delivered set. Legacy external-backend
        modules resolve to ``unimplemented`` automatically.
        """
        return {
            name: cls._modules[name].description
            for name in sorted(cls._modules)
            if (not packaged_only or cls.is_packaged_module(cls._modules[name]))
            and (
                include_external_backends
                or cls._modules[name].resolved_implementation_status() != "unimplemented"
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

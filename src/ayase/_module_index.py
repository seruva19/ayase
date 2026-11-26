"""Read and generate the packaged module index used for named lookups."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, cast


_INDEX_PATH = Path(__file__).with_suffix(".json")


@lru_cache(maxsize=1)
def load_module_index() -> Dict[str, Dict[str, Any]]:
    try:
        return cast(Dict[str, Dict[str, Any]], json.loads(_INDEX_PATH.read_text(encoding="utf-8")))
    except FileNotFoundError:
        # Source checkouts can regenerate the index through full discovery.
        return {}


def generate_module_index() -> Dict[str, Dict[str, Any]]:
    from .pipeline import MODULE_ALIASES, ModuleRegistry

    ModuleRegistry.discover_modules()
    index = {}
    for name in ModuleRegistry.list_modules(packaged_only=True):
        module_class = ModuleRegistry.get_module(name)
        if module_class is None:
            raise RuntimeError(f"Registered module {name!r} disappeared during discovery")
        metadata = module_class.get_metadata()
        index[name] = {
            "import_path": module_class.__module__,
            "class_name": module_class.__qualname__,
            "aliases": sorted(alias for alias, target in MODULE_ALIASES.items() if target == name),
            "provenance": metadata.get("provenance") or {},
            "output_fields": sorted(metadata.get("output_fields") or ()),
            "dataset_output_fields": sorted(metadata.get("dataset_output_fields") or ()),
            "metric_groups": getattr(module_class, "metric_groups", None) or {},
        }
    return index


def write_module_index(path: Path) -> None:
    document = json.dumps(generate_module_index(), ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(document + "\n", encoding="utf-8")

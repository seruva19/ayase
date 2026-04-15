"""Profile loader and module builder for app-specific Ayase pipelines.

This module intentionally does not ship hardcoded metric presets.
Applications define their own profiles (JSON/TOML or dict) and build
module instances from them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import BaseModel, Field

from .config import AyaseConfig
from .pipeline import ModuleRegistry, PipelineModule


class PipelineProfile(BaseModel):
    """External profile schema for assembling Ayase pipelines."""

    name: Optional[str] = None
    modules: List[str] = Field(default_factory=list)
    module_config: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


def load_profile(profile: Union[Path, str, Dict[str, Any], PipelineProfile]) -> PipelineProfile:
    """Load a profile from an object, dict, JSON, or TOML file."""

    if isinstance(profile, PipelineProfile):
        return profile

    if isinstance(profile, dict):
        return cast(PipelineProfile, PipelineProfile.model_validate(profile))

    if isinstance(profile, str):
        profile = Path(profile)

    if not isinstance(profile, Path):
        raise TypeError("profile must be Path, str, dict, or PipelineProfile")

    if not profile.exists():
        raise FileNotFoundError(f"Profile file not found: {profile}")

    suffix = profile.suffix.lower()
    raw = profile.read_text(encoding="utf-8")

    if suffix == ".json":
        data = cast(Dict[str, Any], json.loads(raw))
    elif suffix in (".toml", ".tml"):
        try:
            import tomllib  # py3.11+
        except ModuleNotFoundError:
            import tomli as tomllib  # py3.9/3.10 fallback
        data = cast(Dict[str, Any], tomllib.loads(raw))
    else:
        raise ValueError(f"Unsupported profile format: {suffix}. Use .json or .toml")

    return cast(PipelineProfile, PipelineProfile.model_validate(data))


def instantiate_profile_modules(
    profile: Union[Path, str, Dict[str, Any], PipelineProfile],
    config: Optional[AyaseConfig] = None,
) -> List[PipelineModule]:
    """Instantiate modules for an app-defined profile."""

    loaded = load_profile(profile)
    cfg = config or AyaseConfig.load()

    ModuleRegistry.discover_modules(plugin_paths=cfg.pipeline.plugin_folders)

    if not loaded.modules:
        return []

    modules: List[PipelineModule] = []
    for module_name in loaded.modules:
        module_cls = ModuleRegistry.get_module(module_name)
        if module_cls is None:
            raise ValueError(f"Unknown module in profile: {module_name}")

        params: Dict[str, Any] = {
            "models_dir": str(cfg.general.models_dir),
            "parallel_jobs": cfg.general.parallel_jobs,
        }

        per_module = loaded.module_config.get(module_name, {})
        if per_module:
            params.update(per_module)

        modules.append(module_cls(config=params))

    return modules

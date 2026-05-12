"""Compatibility shims for stale third-party dependencies.

This module is imported very early in ayase/__init__.py so that the patches
take effect before any consumer (pyiqa, ImageReward, etc.) imports the
problematic packages.

Patches:
    clip — OpenAI CLIP package (last released 2021) uses
           `from pkg_resources import packaging`, deprecated in modern
           setuptools. We pre-load our vendored copy (at ayase/_vendor/clip)
           into sys.modules['clip'] so any subsequent `import clip` resolves
           to the patched version.

This module is idempotent: re-importing or repeated invocation is a no-op.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


_PATCHED_FLAG = "_ayase_compat_applied"
_VENDOR_DIR = Path(__file__).parent / "_vendor"


def _install_vendored_package(name: str, vendor_subdir: str) -> None:
    """Register a vendored package in sys.modules under the given name.

    Subsequent `import {name}` anywhere in the process — including inside
    transitively-imported third-party libs — will resolve to the vendored copy
    instead of looking for {name} on the regular sys.path.
    """
    if name in sys.modules:
        return  # an already-loaded module wins; don't second-guess

    pkg_path = _VENDOR_DIR / vendor_subdir
    init_path = pkg_path / "__init__.py"
    if not init_path.exists():
        return  # vendor copy missing — silently skip; fall back to pip-installed

    spec = importlib.util.spec_from_file_location(
        name,
        init_path,
        submodule_search_locations=[str(pkg_path)],
    )
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def apply_patches() -> None:
    """Apply all compatibility patches. Safe to call multiple times."""
    if getattr(sys, _PATCHED_FLAG, False):
        return

    # OpenAI CLIP: vendored copy bypasses deprecated pkg_resources.packaging.
    _install_vendored_package("clip", "clip")

    setattr(sys, _PATCHED_FLAG, True)


apply_patches()

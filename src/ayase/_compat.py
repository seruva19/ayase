"""Compatibility shims for stale third-party dependencies.

This module is imported very early in ayase/__init__.py so that the patches
take effect before any consumer (pyiqa, ImageReward, RAM, etc.) imports the
problematic packages.

Patches:
    clip — OpenAI CLIP package (last released 2021) uses
           `from pkg_resources import packaging`, deprecated in modern
           setuptools. We pre-load our vendored copy (at ayase/_vendor/clip)
           into sys.modules['clip'] so any subsequent `import clip` resolves
           to the patched version.

    transformers.modeling_utils — older downstream libs (RAM bert.py,
           ImageReward BLIP med.py, etc.) still import
           `apply_chunking_to_forward`, `find_pruneable_heads_and_indices`,
           and `prune_linear_layer` from `transformers.modeling_utils`,
           but newer transformers (>=4.31?) moved them to
           `transformers.pytorch_utils`. We re-export them on the old path
           so those libs import without source patches.

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


def _backport_modeling_utils() -> None:
    """Re-export symbols that moved from modeling_utils → pytorch_utils.

    Old libs (RAM bert.py, ImageReward BLIP med.py, etc.) still do
    `from transformers.modeling_utils import apply_chunking_to_forward, ...`
    which fails on newer transformers. We add the symbols back to the
    modeling_utils namespace so the legacy import path resolves.
    """
    try:
        import transformers.modeling_utils as _mu
        import transformers.pytorch_utils as _pu
    except ImportError:
        return
    for sym in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(_mu, sym) and hasattr(_pu, sym):
            setattr(_mu, sym, getattr(_pu, sym))


def apply_patches() -> None:
    """Apply all compatibility patches. Safe to call multiple times."""
    if getattr(sys, _PATCHED_FLAG, False):
        return

    # OpenAI CLIP: vendored copy bypasses deprecated pkg_resources.packaging.
    _install_vendored_package("clip", "clip")

    # transformers.modeling_utils → pytorch_utils symbol relocations.
    _backport_modeling_utils()

    setattr(sys, _PATCHED_FLAG, True)


apply_patches()

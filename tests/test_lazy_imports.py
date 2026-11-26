"""Cold imports and the compatibility patch entry points."""

import os
import subprocess
import sys
from pathlib import Path


def _run_imports(code, *paths):
    source = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (*paths, source))))
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=60)


def test_public_imports_are_lazy():
    _run_imports(
        """
import sys
import ayase
import ayase.modules as modules

assert not {"clip", "torch", "torchvision", "transformers"} & sys.modules.keys()
assert not any(name.startswith("ayase.modules.") for name in sys.modules)
assert "DOVERModule" in dir(modules)
from ayase.modules import DOVERModule
from ayase.modules.dover import DOVERModule as Direct
assert DOVERModule is Direct
assert "ayase.modules.motion" not in sys.modules
"""
    )


def test_vendored_clip_import_preserves_package_and_existing_module():
    _run_imports(
        """
import sys
import types
from pathlib import Path

existing = sys.modules["clip"] = types.ModuleType("clip")
import ayase
import clip
assert clip is existing
del sys.modules["clip"]

import clip
from clip.simple_tokenizer import SimpleTokenizer
assert Path(clip.__file__).parent.parent.name == "_vendor"
assert callable(SimpleTokenizer)
"""
    )


def test_managed_worker_resolves_legacy_transformers_import(tmp_path):
    package = tmp_path / "transformers"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "modeling_utils.py").write_text(
        "def prune_linear_layer(): return 'original'\n", encoding="utf-8"
    )
    (package / "pytorch_utils.py").write_text(
        "def apply_chunking_to_forward(): return 'chunk'\n"
        "def find_pruneable_heads_and_indices(): return 'heads'\n"
        "def prune_linear_layer(): return 'replacement'\n",
        encoding="utf-8",
    )
    (tmp_path / "legacy_metric.py").write_text(
        "from transformers.modeling_utils import (apply_chunking_to_forward,\n"
        "    find_pruneable_heads_and_indices, prune_linear_layer)\n"
        "class LegacyMetric:\n"
        "    values = (apply_chunking_to_forward(), find_pruneable_heads_and_indices(),\n"
        "              prune_linear_layer())\n",
        encoding="utf-8",
    )
    _run_imports(
        """
import sys
from ayase._managed_module_worker import _resolve_class
assert "transformers.modeling_utils" not in sys.modules
metric = _resolve_class("legacy_metric", "LegacyMetric")
assert metric.values == ("chunk", "heads", "original")
""",
        tmp_path,
    )

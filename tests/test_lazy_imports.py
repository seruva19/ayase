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


def test_named_profile_uses_index_and_preserves_dataset_provenance():
    _run_imports(
        """
import sys
import warnings
from ayase.config import AyaseConfig
from ayase.pipeline import ModuleRegistry, Pipeline
from ayase.profile import instantiate_profile_modules

config = AyaseConfig()
config.pipeline.plugin_folders = []
modules = instantiate_profile_modules({"modules": ["dover", "motion", "fid"]}, config)
assert [module.name for module in modules] == ["dover", "motion", "fid"]
assert "ayase.modules.fvd" not in sys.modules
assert "ayase.modules.finevq_raw" not in sys.modules

fid = type(modules[-1])
def fail(*args, **kwargs):
    raise AssertionError("source introspection was called")
fid.get_metadata = fail
assert Pipeline(modules)._dataset_field_provenance["fid"] == "adapted"
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter("always")
    assert ModuleRegistry.get_module("finevq") is ModuleRegistry.get_module("finevq_raw")
    assert any("was renamed" in str(w.message) for w in captured)
assert ModuleRegistry.get_module("lip_sync").__name__ == "LipSyncModule"
"""
    )


def test_named_discovery_rejects_plugin_collision(tmp_path):
    (tmp_path / "duplicate.py").write_text(
        "from ayase.pipeline import PipelineModule\n"
        "class Duplicate(PipelineModule):\n"
        "    name = 'dover'\n"
        "    def process(self, sample): return sample\n",
        encoding="utf-8",
    )
    _run_imports(
        f"""
from pathlib import Path
from ayase.pipeline import ModuleRegistry
folder = Path({str(tmp_path)!r})
ModuleRegistry.prepare_named_lookup([folder])
assert ModuleRegistry.get_module("dover").__module__ == "ayase.modules.dover"
report = ModuleRegistry.readiness_report()
duplicate = report[str(folder / "duplicate.py")]
assert duplicate["status"] == "missing"
assert "Duplicate module name" in duplicate["error"]
"""
    )


def test_packaged_index_matches_full_discovery():
    from ayase._module_index import generate_module_index, load_module_index

    assert load_module_index() == generate_module_index()


def test_failed_indexed_import_rolls_back_partial_registration(tmp_path, monkeypatch):
    from ayase import pipeline

    (tmp_path / "broken_metric.py").write_text(
        "from ayase.pipeline import PipelineModule\n"
        "class Broken(PipelineModule):\n"
        "    name = 'test_indexed_broken'\n"
        "    def process(self, sample): return sample\n"
        "raise ImportError('missing backend')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        pipeline,
        "load_module_index",
        lambda: {
            "test_indexed_broken": {"import_path": "broken_metric", "class_name": "Broken"}
        },
    )
    readiness = dict(pipeline.ModuleRegistry._readiness)
    monkeypatch.setattr(pipeline.ModuleRegistry, "_readiness", readiness)

    assert pipeline.ModuleRegistry.get_module("test_indexed_broken") is None
    assert "test_indexed_broken" not in pipeline.ModuleRegistry._modules
    assert "broken_metric" not in sys.modules
    assert readiness["broken_metric"] == {"status": "missing", "error": "missing backend"}


def test_apply_patches_disables_tf_and_flax_without_overriding(monkeypatch):
    import os
    from ayase import _compat

    for key in ("USE_TF", "USE_FLAX", "TRANSFORMERS_NO_TF", "TRANSFORMERS_NO_FLAX"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(sys, _compat._PATCHED_FLAG, False, raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))

    _compat.apply_patches()
    assert os.environ["USE_TF"] == "0"
    assert os.environ["USE_FLAX"] == "0"
    assert os.environ["TRANSFORMERS_NO_TF"] == "1"
    assert os.environ["TRANSFORMERS_NO_FLAX"] == "1"

    monkeypatch.setenv("USE_TF", "1")
    monkeypatch.setattr(sys, _compat._PATCHED_FLAG, False, raising=False)
    _compat.apply_patches()
    assert os.environ["USE_TF"] == "1"

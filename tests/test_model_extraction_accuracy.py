"""Offline coverage for evidence-based model and checkpoint discovery."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from ayase.pipeline import PipelineModule

_MODEL_SOURCE_GLOBAL = "example/module-model"


class _CatalogLoader:
    @staticmethod
    def from_pretrained(model):
        return model


class _ModelSourceFixture(PipelineModule):
    DEFAULT_MODEL = "example/class-model"
    CHECKPOINT_URL = "https://huggingface.co/example/own/resolve/pinned/head.bin"

    def setup(self):
        _CatalogLoader.from_pretrained(_MODEL_SOURCE_GLOBAL)
        _CatalogLoader.from_pretrained(self.DEFAULT_MODEL)

    def process(self, sample):
        return sample


class _SiblingSourceFixture(PipelineModule):
    CHECKPOINT_URL = "https://huggingface.co/example/sibling/resolve/main/unrelated.bin"

    def setup(self):
        _CatalogLoader.from_pretrained("example/sibling-model")

    def process(self, sample):
        return sample


def test_model_analysis_source_retains_class_and_global_scope_without_sibling_models():
    from ayase.models_doc import (
        _extract_hf_checkpoint_urls,
        _extract_hf_models,
        _get_model_analysis_source,
        _get_module_source,
        _get_source,
    )

    source = _get_model_analysis_source(_ModelSourceFixture)
    assert _extract_hf_models(source) == ["example/class-model", "example/module-model"]
    assert _extract_hf_checkpoint_urls(source) == [_ModelSourceFixture.CHECKPOINT_URL]
    assert _get_source(_ModelSourceFixture) == source
    assert _get_module_source(_ModelSourceFixture) == source


def test_unqualified_method_names_skip_class_namespace_but_self_uses_class_default():
    from ayase.models_doc import _extract_hf_models

    source = """
MODEL = "example/global"
class Module:
    MODEL = "example/class"
    def setup(self):
        Loader.from_pretrained(MODEL)
        Loader.from_pretrained(self.MODEL)
"""
    assert _extract_hf_models(source) == ["example/class", "example/global"]


def test_hf_sequential_reassignment_replaces_old_values_at_each_alias_use():
    from ayase.models_doc import _extract_hf_models

    source = """
def setup():
    name = "example/not-loaded"
    name = "example/loaded"
    alias = name
    name = "example/later-unused"
    Loader.from_pretrained(alias)
"""
    assert _extract_hf_models(source) == ["example/loaded"]


def test_hf_branch_join_keeps_possible_models_but_unconditional_overwrite_replaces_them():
    from ayase.models_doc import _extract_hf_models

    source = """
def setup(flag):
    name = "example/never-loaded"
    if flag:
        name = "example/branch-one"
    else:
        name = "example/branch-two"
    Loader.from_pretrained(name)
    name = "example/after-branches"
    Loader.from_pretrained(name)
"""
    assert _extract_hf_models(source) == [
        "example/after-branches",
        "example/branch-one",
        "example/branch-two",
    ]


def test_hf_discovery_ignores_prose_unused_config_and_other_function_locals():
    from ayase.models_doc import _extract_hf_models

    source = '''
"""AutoModel.from_pretrained("example/documentation")"""
# AutoModel.from_pretrained("example/comment")
source_file = "anyaudio_judge/judge.py"
adapter_path = "lora_tuned/checkpoint-550"
def unrelated():
    model_name = "example/unused"
def load():
    model_name = "Qwen/Qwen2.5-Omni-7B"
    return AutoModel.from_pretrained(model_name)
'''
    assert _extract_hf_models(source, {"model_name": "example/unused-config"}) == [
        "Qwen/Qwen2.5-Omni-7B"
    ]


def test_hf_discovery_traces_used_config_defaults_aliases_and_project_wrappers():
    from ayase.models_doc import _extract_hf_models

    source = """
class Module:
    def __init__(self):
        self.model = self.config.get("backbone", "example/fallback")
    def setup(self):
        path = resolve_model_path(self.model, "models")
        alias = path
        return from_pretrained_with_attention(AutoModel, alias, self.config)
    def processor(self):
        return AutoProcessor.from_pretrained(
            pretrained_model_name_or_path="example/processor.v2")
    def auxiliary(self):
        return from_pretrained_with_attention(
            model_cls=AutoModel, model_id="example/auxiliary.v1", config=self.config)
"""
    assert _extract_hf_models(source, {"backbone": "example/configured.v3"}) == [
        "example/auxiliary.v1",
        "example/configured.v3",
        "example/processor.v2",
    ]


def test_hf_discovery_handles_cycles_and_snapshot_downloads_without_execution():
    from ayase.models_doc import _extract_hf_models

    source = """
loop = loop
first = second
second = first
AutoModel.from_pretrained(loop)
AutoModel.from_pretrained(first)
root = download_hf_snapshot("example/weights.v1", "models")
AutoModel.from_pretrained(str(root))
"""
    assert _extract_hf_models(source) == ["example/weights.v1"]


def test_hf_discovery_resolves_class_defaults_but_does_not_catalog_dataset_repos():
    from ayase.models_doc import _extract_hf_models

    source = """
class Module:
    model_id = "microsoft/wavlm-large"
    def setup(self):
        snapshot_download(repo_id=self.model_id)
        snapshot_download(repo_id="example/evaluation-set", repo_type="dataset")
"""
    assert _extract_hf_models(source) == ["microsoft/wavlm-large"]


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        (
            "action_recognition",
            {"MCG-NJU/videomae-large-finetuned-kinetics", "openai/clip-vit-base-patch32"},
        ),
        ("camerabench", {"Qwen/Qwen2.5-VL-7B-Instruct", "chancharikm/qwen2.5-vl-7b-cam-motion"}),
        (
            "dice_edit",
            {
                "HuggingFaceM4/Idefics3-8B-Llama3",
                "aimagelab/DICE_coherence_Idefics",
                "aimagelab/DICE_differencedet_Idefics",
            },
        ),
    ],
)
def test_real_module_loader_arguments_preserve_models_and_exclude_local_paths(module, expected):
    from ayase.models_doc import _extract_hf_models

    path = Path(__file__).parents[1] / "src" / "ayase" / "modules" / f"{module}.py"
    assert set(_extract_hf_models(path.read_text(encoding="utf-8"))) == expected


def test_checkpoint_discovery_preserves_revision_dotted_repo_and_literal_concatenation():
    from ayase.models_doc import _extract_hf_checkpoint_urls

    source = '''
"""https://huggingface.co/example/prose/resolve/main/example.bin"""
# https://huggingface.co/example/comment/resolve/main/example.bin
URL = ('https://huggingface.co/example/model.v2/resolve/'
       'abc123/nested/head.bin?download=true')
CARD = "https://huggingface.co/example/model.v2/blob/v1/head.bin"
DYNAMIC = f"https://huggingface.co/example/model.v2/resolve/main/nested/{filename}"
'''
    assert _extract_hf_checkpoint_urls(source) == [
        "https://huggingface.co/example/model.v2/blob/v1/head.bin",
        "https://huggingface.co/example/model.v2/resolve/abc123/nested/head.bin?download=true",
    ]


def test_generated_checkpoint_links_do_not_replace_pinned_revision_with_main(monkeypatch):
    from ayase import models_doc

    url = "https://huggingface.co/example/model.v2/resolve/abc123/nested/head.bin"
    module = SimpleNamespace(name="checkpoint_fixture", default_config={}, models=[])
    monkeypatch.setattr(models_doc.ModuleRegistry, "discover_modules", lambda: None)
    monkeypatch.setattr(models_doc.ModuleRegistry, "list_modules", lambda **kw: [module.name])
    monkeypatch.setattr(models_doc.ModuleRegistry, "get_module", lambda name: module)
    monkeypatch.setattr(models_doc, "_get_source", lambda cls: "class Module: pass")
    monkeypatch.setattr(models_doc, "_get_module_source", lambda cls: f"URL = {url!r}")
    monkeypatch.setattr(
        models_doc, "compute_external_backend_partition", lambda mods: (set(), set())
    )
    monkeypatch.setattr(models_doc, "_generate_charts", lambda *args: {})

    output = models_doc.generate_models_doc(fetch_licenses=False)
    assert f"]({url})" in output
    assert "/resolve/main/nested/head.bin" not in output

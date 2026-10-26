"""Offline checks for model identity, useful references and consistent catalog inputs."""

from types import SimpleNamespace

from ayase.metric_catalog import ModelCatalogItem, _collect_models, _model_key
from ayase.pipeline import PipelineModule


def _models(monkeypatch, source, declarations=()):
    monkeypatch.setattr("ayase.metric_catalog._get_model_analysis_source", lambda cls: source)
    cls = SimpleNamespace(name="reference_fixture", description="Reference fixture scoring")
    return _collect_models(cls, {"models": list(declarations)}, source)


def test_provider_references_do_not_merge_distinct_metrics(monkeypatch):
    models = _models(
        monkeypatch,
        'one = pyiqa.create_metric("ssim")\ntwo = pyiqa.create_metric("psnr")',
    )
    assert {model.name for model in models} == {"pyiqa/ssim", "pyiqa/psnr"}
    assert all(model.url == "https://github.com/chaofengc/IQA-PyTorch" for model in models)
    assert all(model.task for model in models)


def test_explicit_and_detected_provider_model_share_identity():
    explicit = ModelCatalogItem(name="ssim", source="pyiqa")
    detected = ModelCatalogItem(
        name="pyiqa/ssim", source="pyiqa", url="https://github.com/chaofengc/IQA-PyTorch"
    )
    assert _model_key(explicit) == _model_key(detected)


def test_checkpoint_dedup_keeps_revision_file_and_declared_purpose(monkeypatch):
    root = "https://huggingface.co/owner.with-dot/model.repo"
    source = (
        f'first = "{root}/resolve/pinned/head.bin"\n'
        f'second = "{root}/resolve/other/head.bin"\n'
        f'third = "{root}/resolve/pinned/encoder.bin"\n'
    )
    models = _models(
        monkeypatch,
        source,
        [
            {
                "id": "owner.with-dot/model.repo",
                "type": "huggingface",
                "url": root + "/blob/pinned/head.bin",
                "task": "Trained quality head",
            }
        ],
    )
    assert len(models) == 3
    head = next(model for model in models if model.url.endswith("/blob/pinned/head.bin"))
    assert head.name == "head.bin"
    assert head.task == "Trained quality head"
    assert any(model.url.endswith("/resolve/other/head.bin") for model in models)
    assert any(model.url.endswith("/resolve/pinned/encoder.bin") for model in models)


def test_provider_urls_and_context_are_available_without_network(monkeypatch):
    models = _models(monkeypatch, 'encoder = torch.hub.load("owner/repo:release", "encoder")')
    assert len(models) == 1
    assert models[0].url == "https://github.com/owner/repo/tree/release"
    assert models[0].task == "Used by reference_fixture: Reference fixture scoring"


def test_checkpoint_keeps_explicit_local_asset_name(monkeypatch):
    url = "https://huggingface.co/owner/repo/resolve/pinned/remote-head.bin"
    models = _models(
        monkeypatch,
        f'url = "{url}"',
        [{"id": "local-head.pt", "type": "local", "url": url, "task": "Quality regression head"}],
    )
    assert len(models) == 1
    assert models[0].name == "local-head.pt"
    assert models[0].task == "Quality regression head"


class _ExplicitInput(PipelineModule):
    """Image input; prose mentions reference_path and sample.caption only as examples."""

    input_type = "img"

    def process(self, sample):
        return sample


def test_fast_and_detailed_catalog_share_authoritative_input_type():
    from ayase.metric_catalog import _build_module_item

    for detailed in (False, True):
        item = _build_module_item(_ExplicitInput, detailed=detailed)
        assert item.input_type == "img"


def test_cli_renders_provider_link_and_install_reference(monkeypatch):
    from rich.console import Console
    from ayase.metric_catalog import _build_module_item, _render_module

    item = _build_module_item(_ExplicitInput, detailed=False)
    model = ModelCatalogItem(
        name="pyiqa/ssim",
        source="pyiqa",
        task="Structural similarity",
        url="https://github.com/chaofengc/IQA-PyTorch",
        install="pip install pyiqa",
    )
    from dataclasses import replace

    console = Console(record=True, width=240)
    _render_module(console, replace(item, models=(model,)))
    output = console.export_text()
    assert "Structural similarity" in output
    assert "https://github.com/chaofengc/IQA-PyTorch" in output
    assert "pip install pyiqa" in output

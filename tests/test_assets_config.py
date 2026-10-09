"""Focused contracts for configurable Ayase asset repositories."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ayase.config import (
    DEFAULT_ASSETS_REPO,
    AyaseConfig,
    GeneralConfig,
    resolve_assets_repo,
    resolve_assets_url,
)
from ayase.profile import PipelineProfile, instantiate_profile_modules
from ayase.runtime import runtime_module_config


def test_assets_repo_defaults_and_normalizes_hugging_face_url():
    assert GeneralConfig().assets_repo == DEFAULT_ASSETS_REPO
    assert (
        GeneralConfig(assets_repo="https://huggingface.co/example-org/example-assets/").assets_repo
        == "example-org/example-assets"
    )


def test_assets_repo_loads_from_file_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "ayase.toml"
    config_path.write_text('[general]\nassets_repo = "file-owner/file-assets"\n', encoding="utf-8")
    assert AyaseConfig.load(config_path).general.assets_repo == "file-owner/file-assets"

    monkeypatch.setenv("AYASE_GENERAL__ASSETS_REPO", "env-owner/env-assets")
    assert AyaseConfig.load(config_path).general.assets_repo == "env-owner/env-assets"


@pytest.mark.parametrize(
    "value",
    [
        "owner/repo/file.bin",
        "owner/../repo",
        "owner/repo..name",
        "owner/repo--name",
        "owner/-repo",
        "owner/repo.",
        "owner/repo?token=secret",
        "owner/repo#fragment",
        "https://example.com/owner/repo",
        "http://huggingface.co/owner/repo",
        "https://huggingface.co/owner/repo/resolve/main/file.bin",
        "https://token@huggingface.co/owner/repo",
    ],
)
def test_assets_repo_rejects_non_repository_inputs(value: str):
    with pytest.raises(ValidationError):
        GeneralConfig(assets_repo=value)


def test_runtime_injection_and_profile_module_override():
    config = AyaseConfig(general=GeneralConfig(assets_repo="global/assets"))
    assert runtime_module_config(config)["assets_repo"] == "global/assets"

    modules = instantiate_profile_modules(
        PipelineProfile(
            modules=["metadata"],
            module_config={"metadata": {"assets_repo": "module/assets"}},
        ),
        config,
    )
    assert modules[0].config["assets_repo"] == "module/assets"


def test_resolve_assets_repo_accepts_id_and_url():
    assert resolve_assets_repo() == DEFAULT_ASSETS_REPO
    assert resolve_assets_repo({"assets_repo": "owner/repo"}) == "owner/repo"
    assert (
        resolve_assets_repo({"assets_repo": "https://huggingface.co/owner/repository"})
        == "owner/repository"
    )


def test_resolve_assets_url_preserves_revision_filename_path_and_query():
    source = (
        "https://huggingface.co/AkaneTendo25/ayase-assets/resolve/"
        "refs%2Fpr%2F7/checkpoints/nested/model.pt?download=true"
    )
    assert resolve_assets_url(source, {"assets_repo": "mirror/assets"}) == (
        "https://huggingface.co/mirror/assets/resolve/"
        "refs%2Fpr%2F7/checkpoints/nested/model.pt?download=true"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/another-owner/ayase-assets/resolve/main/model.pt",
        "https://huggingface.co/AkaneTendo25/another-repo/resolve/main/model.pt",
        "https://example.com/AkaneTendo25/ayase-assets/resolve/main/model.pt",
    ],
)
def test_resolve_assets_url_leaves_unrelated_urls_unchanged(url: str):
    assert resolve_assets_url(url, {"assets_repo": "mirror/assets"}) == url

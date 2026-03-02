from pathlib import Path

import pytest

from ayase.config import AyaseConfig
from ayase.profile import PipelineProfile, instantiate_profile_modules, load_profile


def test_load_profile_from_dict():
    profile = load_profile(
        {
            "name": "app_profile",
            "modules": ["metadata", "basic_quality"],
            "module_config": {"metadata": {"sample_rate": 4}},
        }
    )
    assert isinstance(profile, PipelineProfile)
    assert profile.name == "app_profile"
    assert profile.modules == ["metadata", "basic_quality"]
    assert profile.module_config["metadata"]["sample_rate"] == 4


def test_load_profile_from_json(tmp_path: Path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        (
            "{"
            '"name":"json_profile",'
            '"modules":["metadata","basic_quality"],'
            '"module_config":{"metadata":{"sample_rate":2}}'
            "}"
        ),
        encoding="utf-8")
    profile = load_profile(profile_path)
    assert profile.name == "json_profile"
    assert profile.modules == ["metadata", "basic_quality"]
    assert profile.module_config["metadata"]["sample_rate"] == 2


def test_load_profile_from_toml(tmp_path: Path):
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        (
            'name = "toml_profile"\n'
            'modules = ["metadata", "basic_quality"]\n'
            "\n"
            "[module_config.metadata]\n"
            "sample_rate = 3\n"
        ),
        encoding="utf-8")
    profile = load_profile(profile_path)
    assert profile.name == "toml_profile"
    assert profile.modules == ["metadata", "basic_quality"]
    assert profile.module_config["metadata"]["sample_rate"] == 3


def test_instantiate_profile_modules():
    config = AyaseConfig()
    modules = instantiate_profile_modules(
        {
            "modules": ["metadata", "basic_quality"],
            "module_config": {"metadata": {"sample_rate": 5}},
        },
        config=config)
    assert len(modules) == 2
    assert modules[0].name == "metadata"
    assert modules[1].name == "basic_quality"
    assert modules[0].config["sample_rate"] == 5


def test_instantiate_profile_modules_unknown_module():
    config = AyaseConfig()
    with pytest.raises(ValueError, match="Unknown module in profile"):
        instantiate_profile_modules({"modules": ["no_such_module"]}, config=config)

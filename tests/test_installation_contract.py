"""The ordinary Ayase install exposes every metric without per-metric extras."""

from pathlib import Path

from packaging.requirements import Requirement


def _project():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_no_optional_runtime_dependencies():
    config = _project()
    assert not config["project"].get("optional-dependencies")
    dependencies = {
        Requirement(spec).name.lower().replace("_", "-")
        for spec in config["project"]["dependencies"]
    }
    assert {"sentencepiece", "cdpam", "resampy", "uv", "zipfile-deflate64"} <= dependencies


def test_development_tools_are_separate_from_metric_install():
    config = _project()
    development = config["dependency-groups"]["dev"]
    assert any(Requirement(spec).name == "pytest" for spec in development)
    assert all("cdpam" not in spec and "sentencepiece" not in spec for spec in development)

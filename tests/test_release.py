from pathlib import Path


def _write_release_fixture(root: Path) -> None:
    (root / "src" / "ayase").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "ayase"\nversion = "0.1.52"\n',
        encoding="utf-8",
    )
    (root / "src" / "ayase" / "__init__.py").write_text(
        '__version__ = "0.1.52"\n',
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "### Added\n\n"
        "- **demo**: added a release note.\n\n"
        "## [0.1.52]\n\n"
        "### Fixed\n\n"
        "- Existing release.\n",
        encoding="utf-8",
    )


def test_prepare_release_bumps_versions_and_promotes_changelog(tmp_path):
    from ayase.release import prepare_release

    _write_release_fixture(tmp_path)

    result = prepare_release(
        "0.1.53",
        root=tmp_path,
        release_date="2026-05-28",
        regenerate_docs=False,
    )

    assert result.pyproject_changed
    assert result.init_changed
    assert result.changelog_changed
    assert 'version = "0.1.53"' in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert '__version__ = "0.1.53"' in (
        tmp_path / "src" / "ayase" / "__init__.py"
    ).read_text(encoding="utf-8")

    changelog = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]\n\n## [0.1.53] - 2026-05-28" in changelog
    assert "- **demo**: added a release note." in changelog


def test_promote_changelog_empty_unreleased_is_idempotent(tmp_path):
    from ayase.release import promote_changelog

    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [0.1.52]\n\n- Existing release.\n",
        encoding="utf-8",
    )

    assert not promote_changelog(changelog, "0.1.53", "2026-05-28")
    assert "## [0.1.53]" not in changelog.read_text(encoding="utf-8")

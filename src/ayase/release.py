"""Release preparation helpers.

This module keeps the manual release checklist executable: bump both version
locations, promote the curated changelog entries, and refresh generated docs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional


_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[a-zA-Z0-9_.!+\-]*)?$")


@dataclass
class ReadmeSyncResult:
    changed: bool
    modules: int
    metrics: int
    categories: int


@dataclass
class ReleasePrepResult:
    version: str
    pyproject_changed: bool
    init_changed: bool
    changelog_changed: bool
    metrics_written: bool
    models_written: bool
    readme: Optional[ReadmeSyncResult]


def validate_version(version: str) -> str:
    """Return a normalized version string or raise ValueError."""

    normalized = version.strip()
    if not _VERSION_RE.match(normalized):
        raise ValueError(f"Invalid release version: {version!r}")
    return normalized


def bump_version_files(root: Path, version: str) -> tuple[bool, bool]:
    """Update pyproject.toml and src/ayase/__init__.py to *version*."""

    version = validate_version(version)
    pyproject = root / "pyproject.toml"
    init_file = root / "src" / "ayase" / "__init__.py"

    pyproject_changed = _replace_once(
        pyproject,
        r'(?m)^version = "[^"]+"',
        f'version = "{version}"',
    )
    init_changed = _replace_once(
        init_file,
        r'(?m)^__version__ = "[^"]+"',
        f'__version__ = "{version}"',
    )
    return pyproject_changed, init_changed


def promote_changelog(
    changelog_path: Path,
    version: str,
    release_date: Optional[str] = None,
) -> bool:
    """Move current ``[Unreleased]`` entries under a concrete version header.

    Returns True when the changelog was modified. Empty ``[Unreleased]``
    sections are left alone so repeated release prep is idempotent.
    """

    version = validate_version(version)
    release_date = release_date or date.today().isoformat()
    text = changelog_path.read_text(encoding="utf-8")

    if re.search(rf"(?m)^## \[{re.escape(version)}\](?:\s|-|$)", text):
        raise ValueError(f"CHANGELOG.md already contains a section for {version}")

    match = re.search(r"(?m)^## \[Unreleased\]\s*$", text)
    if not match:
        raise ValueError("CHANGELOG.md is missing a `## [Unreleased]` section")

    next_match = re.search(r"(?m)^## \[[^\]]+\]", text[match.end() :])
    section_start = match.end()
    section_end = match.end() + next_match.start() if next_match else len(text)
    unreleased_body = text[section_start:section_end]
    body = unreleased_body.strip()
    if not body:
        return False

    before = text[: match.end()].rstrip()
    after = text[section_end:].lstrip("\n")
    release_section = f"## [{version}] - {release_date}\n\n{body}"
    new_text = f"{before}\n\n{release_section}\n\n{after}"
    changelog_path.write_text(new_text, encoding="utf-8")
    return True


def sync_readme_counts(readme_path: Path = Path("README.md")) -> ReadmeSyncResult:
    """Update README module/metric/category counts from the packaged registry."""

    from .metrics_doc import _get_quality_metrics_fields
    from .models import QualityMetrics
    from .pipeline import ModuleRegistry

    ModuleRegistry.discover_modules()
    all_modules = ModuleRegistry.list_modules(packaged_only=True)
    total = len([n for n in all_modules if ModuleRegistry.get_module(n) is not None])
    n_fields = len(QualityMetrics.model_fields)

    qm_fields = _get_quality_metrics_fields()
    field_writers: set[str] = set()
    has_no_output_modules = False
    for name in all_modules:
        cls = ModuleRegistry.get_module(name)
        if cls is None:
            continue
        meta = cls.get_metadata()
        if not meta.get("output_fields") and not meta.get("dataset_output_fields"):
            has_no_output_modules = True
        for fn in meta.get("output_fields", {}):
            field_writers.add(fn)

    has_dataset_outputs = any(
        ModuleRegistry.get_module(name) is not None
        and ModuleRegistry.get_module(name).get_metadata().get("dataset_output_fields")
        for name in all_modules
    )
    rendered_cats = {qm_fields[fn]["group"] for fn in field_writers if fn in qm_fields}
    n_categories = (
        len(rendered_cats)
        + (1 if has_dataset_outputs else 0)
        + (1 if has_no_output_modules else 0)
    )

    text = readme_path.read_text(encoding="utf-8")
    replacements = [
        (
            r"\*\*\d+ modules\*\*,\s*\*\*\d+ quality metrics\*\*",
            f"**{total} modules**, **{n_fields} quality metrics**",
        ),
        (
            r"\b\d+ modules produce \d+ metrics across \d+ categories\b",
            f"{total} modules produce {n_fields} metrics across {n_categories} categories",
        ),
        (r"show all \d+ modules", f"show all {total} modules"),
    ]
    new_text = text
    for pattern, repl in replacements:
        new_text = re.sub(pattern, repl, new_text)

    changed = new_text != text
    if changed:
        readme_path.write_text(new_text, encoding="utf-8")
    return ReadmeSyncResult(changed, total, n_fields, n_categories)


def prepare_release(
    version: str,
    root: Path = Path("."),
    release_date: Optional[str] = None,
    regenerate_docs: bool = True,
    run_doc_tests: bool = False,
    fetch_licenses: bool = False,
) -> ReleasePrepResult:
    """Prepare a release by updating version, changelog, docs, and README."""

    version = validate_version(version)
    root = root.resolve()

    pyproject_changed, init_changed = bump_version_files(root, version)
    changelog_changed = promote_changelog(root / "CHANGELOG.md", version, release_date)

    metrics_written = False
    models_written = False
    readme_result: Optional[ReadmeSyncResult] = None

    if regenerate_docs:
        from .metrics_doc import generate_metrics_doc
        from .models_doc import generate_models_doc
        from .pipeline import ModuleRegistry

        ModuleRegistry.discover_modules()
        (root / "METRICS.md").write_text(
            generate_metrics_doc(run_tests=run_doc_tests),
            encoding="utf-8",
        )
        metrics_written = True
        (root / "MODELS.md").write_text(
            generate_models_doc(fetch_licenses=fetch_licenses),
            encoding="utf-8",
        )
        models_written = True
        readme_result = sync_readme_counts(root / "README.md")

    return ReleasePrepResult(
        version=version,
        pyproject_changed=pyproject_changed,
        init_changed=init_changed,
        changelog_changed=changelog_changed,
        metrics_written=metrics_written,
        models_written=models_written,
        readme=readme_result,
    )


def _replace_once(path: Path, pattern: str, replacement: str) -> bool:
    text = path.read_text(encoding="utf-8")
    new_text, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise ValueError(f"Could not update {path}: pattern not found")
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True

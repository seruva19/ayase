"""Consistency tests for the QualityMetrics._FIELD_GROUPS registry.

Verifies that:
- Every QualityMetrics field has a group entry (no silent "other" fallthrough).
- Every key in _FIELD_GROUPS corresponds to a real model field.
- Every key in each module's metric_info/metric_groups maps to a real field.
- Each module's metric_groups declaration actually took effect after discovery.
- register_field_groups() merges entries correctly (unit test, no real modules).
"""

import pytest

from ayase.models import DatasetStats, QualityMetrics
from ayase.pipeline import ModuleRegistry

# ---------------------------------------------------------------------------
# One-time discovery (process-global; safe to call multiple times)
# ---------------------------------------------------------------------------

_discovered = False


def _ensure_discovered() -> None:
    global _discovered
    if not _discovered:
        ModuleRegistry.discover_modules()
        _discovered = True


@pytest.fixture(scope="module", autouse=True)
def discover_once():
    """Run module discovery once for the entire test module."""
    _ensure_discovered()


# ---------------------------------------------------------------------------
# Helper: union of all valid field names
# ---------------------------------------------------------------------------

def _all_valid_fields() -> set:
    return set(QualityMetrics.model_fields.keys()) | set(DatasetStats.model_fields.keys())


# ---------------------------------------------------------------------------
# Test 1: every QualityMetrics field has a group
# ---------------------------------------------------------------------------

def test_every_quality_metric_is_grouped():
    """Every field in QualityMetrics.model_fields must have an entry in _FIELD_GROUPS.

    A missing entry causes to_grouped_dict() to silently bin the metric under
    "other", making it invisible in group-aware views.  When this test fails it
    lists the ungrouped fields so the developer knows exactly what to fix.
    """
    _ensure_discovered()
    all_fields = set(QualityMetrics.model_fields.keys())
    grouped = set(QualityMetrics._FIELD_GROUPS.keys())
    ungrouped = all_fields - grouped
    assert not ungrouped, (
        f"{len(ungrouped)} QualityMetrics field(s) have no entry in _FIELD_GROUPS "
        f"(will fall through to 'other' in to_grouped_dict):\n  "
        + "\n  ".join(sorted(ungrouped))
    )


# ---------------------------------------------------------------------------
# Test 2: every key in _FIELD_GROUPS is a real field
# ---------------------------------------------------------------------------

def test_field_group_keys_are_real_fields():
    """Every key in QualityMetrics._FIELD_GROUPS must be a real model field.

    Stale entries (from renamed/removed fields) indicate a registry that has
    drifted from the model definition.
    """
    _ensure_discovered()
    valid = _all_valid_fields()
    stray = set(QualityMetrics._FIELD_GROUPS.keys()) - valid
    assert not stray, (
        f"{len(stray)} key(s) in _FIELD_GROUPS are not fields of QualityMetrics "
        f"or DatasetStats:\n  " + "\n  ".join(sorted(stray))
    )


# ---------------------------------------------------------------------------
# Test 3: every module's metric_info keys are real fields
# ---------------------------------------------------------------------------

def test_metric_info_keys_are_real_fields():
    """Every key in a module's metric_info must be a real QualityMetrics/DatasetStats field."""
    _ensure_discovered()
    valid = _all_valid_fields()
    problems: list[str] = []
    for mod_name, mod_cls in ModuleRegistry._modules.items():
        info = getattr(mod_cls, "metric_info", None)
        if not info:
            continue
        bad = set(info.keys()) - valid
        if bad:
            problems.append(
                f"  module '{mod_name}': unknown keys: {sorted(bad)}"
            )
    assert not problems, (
        f"{len(problems)} module(s) have metric_info keys that are not real fields:\n"
        + "\n".join(problems)
    )


# ---------------------------------------------------------------------------
# Test 4: every module's metric_groups keys are real fields
# ---------------------------------------------------------------------------

def test_metric_groups_keys_are_real_fields():
    """Every key in a module's metric_groups must be a real QualityMetrics/DatasetStats field."""
    _ensure_discovered()
    valid = _all_valid_fields()
    problems: list[str] = []
    for mod_name, mod_cls in ModuleRegistry._modules.items():
        groups = getattr(mod_cls, "metric_groups", None)
        if not groups:
            continue
        bad = set(groups.keys()) - valid
        if bad:
            problems.append(
                f"  module '{mod_name}': unknown keys: {sorted(bad)}"
            )
    assert not problems, (
        f"{len(problems)} module(s) have metric_groups keys that are not real fields:\n"
        + "\n".join(problems)
    )


# ---------------------------------------------------------------------------
# Test 5: module metric_groups actually took effect in _FIELD_GROUPS
# ---------------------------------------------------------------------------

def test_module_metric_groups_resolve():
    """After discovery, each module's declared metric_groups must be reflected in _FIELD_GROUPS.

    This catches a class of bugs where a module declares groups but they never
    get merged (e.g. because discovery is skipped or register_field_groups has
    a bug).
    """
    _ensure_discovered()
    mismatches: list[str] = []
    for mod_name, mod_cls in ModuleRegistry._modules.items():
        groups = getattr(mod_cls, "metric_groups", None)
        if not groups:
            continue
        for field, expected_group in groups.items():
            actual_group = QualityMetrics._FIELD_GROUPS.get(field)
            if actual_group != expected_group:
                mismatches.append(
                    f"  module='{mod_name}' field='{field}' "
                    f"expected='{expected_group}' actual='{actual_group}'"
                )
    assert not mismatches, (
        f"{len(mismatches)} metric_groups declaration(s) did not take effect:\n"
        + "\n".join(mismatches)
    )


# ---------------------------------------------------------------------------
# Test 6: register_field_groups() merges and overrides correctly (unit test)
# ---------------------------------------------------------------------------

def test_register_field_groups_merges_and_overrides():
    """register_field_groups() must merge new entries and allow overrides.

    This is an isolated unit test: it uses a sentinel key that cannot clash
    with any real field, then restores the registry to its original state via
    a finally block.
    """
    sentinel = "__pytest_tmp_field__"
    original = dict(QualityMetrics._FIELD_GROUPS)
    try:
        # The sentinel must not exist before we add it
        assert sentinel not in QualityMetrics._FIELD_GROUPS, (
            f"Sentinel key '{sentinel}' unexpectedly already in _FIELD_GROUPS"
        )

        # Merge a new entry
        QualityMetrics.register_field_groups({sentinel: "nr_quality"})
        assert QualityMetrics._FIELD_GROUPS.get(sentinel) == "nr_quality", (
            f"After register_field_groups, '{sentinel}' should map to 'nr_quality'"
        )

        # Override with a different group (should update the mapping)
        QualityMetrics.register_field_groups({sentinel: "aesthetic"})
        assert QualityMetrics._FIELD_GROUPS.get(sentinel) == "aesthetic", (
            f"After second register_field_groups call, '{sentinel}' should map to 'aesthetic'"
        )
    finally:
        # Restore original state in-place (never rebind the ClassVar)
        QualityMetrics._FIELD_GROUPS.clear()
        QualityMetrics._FIELD_GROUPS.update(original)

    # Verify restoration
    assert sentinel not in QualityMetrics._FIELD_GROUPS, (
        "Sentinel key was not removed during cleanup — state leaked into other tests"
    )

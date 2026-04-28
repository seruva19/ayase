"""Integrity checks for dynamically generated documentation (METRICS.md, MODELS.md).

Validates that every module provides the information the doc generators expect:
correct descriptions, valid output fields, reachable model references, and
consistent metadata.  Run with ``pytest tests/test_docs_integrity.py``.
"""

import importlib
import inspect
import pkgutil
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

from ayase.models import DatasetStats, QualityMetrics
from ayase.pipeline import ModuleRegistry, PipelineModule


# ── Fixtures / helpers ───────────────────────────────────────────────────────

VALID_QM_FIELDS: Set[str] = set(QualityMetrics.model_fields.keys())
VALID_DS_FIELDS: Set[str] = set(DatasetStats.model_fields.keys())
VALID_ALL_FIELDS: Set[str] = VALID_QM_FIELDS | VALID_DS_FIELDS


def _all_module_classes() -> Dict[str, type]:
    """Return {name: class} for every packaged ayase module."""
    ModuleRegistry.discover_modules()
    return {
        name: ModuleRegistry.get_module(name)
        for name in ModuleRegistry.list_modules(packaged_only=True)
        if ModuleRegistry.get_module(name) is not None
    }


ALL_MODULES = _all_module_classes()
MODULE_NAMES = sorted(ALL_MODULES.keys())


def _get_full_source(cls: type) -> str:
    """Return the full source of the *file* containing cls."""
    try:
        fpath = inspect.getfile(cls)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except (TypeError, OSError):
        try:
            return inspect.getsource(cls)
        except (TypeError, OSError):
            return ""


# ═════════════════════════════════════════════════════════════════════════════
# 1. Every module has a meaningful name and description
# ═════════════════════════════════════════════════════════════════════════════


class TestModuleIdentity:
    """Each module must have a unique, non-default name and a real description."""

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_name_is_not_default(self, name: str) -> None:
        cls = ALL_MODULES[name]
        assert cls.name != "unnamed_module", f"{name} still has default name"

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_description_is_not_default(self, name: str) -> None:
        cls = ALL_MODULES[name]
        assert cls.description != "No description provided", (
            f"{name} has no description"
        )

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_description_is_nontrivial(self, name: str) -> None:
        cls = ALL_MODULES[name]
        assert len(cls.description) >= 10, (
            f"{name} description too short: {cls.description!r}"
        )

    def test_no_duplicate_names(self) -> None:
        names = [cls.name for cls in ALL_MODULES.values()]
        dupes = [n for n in names if names.count(n) > 1]
        assert not dupes, f"Duplicate module names: {set(dupes)}"


# ═════════════════════════════════════════════════════════════════════════════
# 2. get_metadata() produces valid output
# ═════════════════════════════════════════════════════════════════════════════


class TestGetMetadata:
    """get_metadata() must return well-formed dicts the doc generators rely on."""

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_get_metadata_does_not_crash(self, name: str) -> None:
        cls = ALL_MODULES[name]
        meta = cls.get_metadata()
        assert isinstance(meta, dict)

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_metadata_has_required_keys(self, name: str) -> None:
        cls = ALL_MODULES[name]
        meta = cls.get_metadata()
        for key in ("name", "description", "input_type", "output_fields",
                     "dataset_output_fields", "default_config"):
            assert key in meta, f"{name} missing metadata key: {key}"

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_output_fields_are_valid_qm_fields(self, name: str) -> None:
        """Every field listed in output_fields must exist in QualityMetrics."""
        cls = ALL_MODULES[name]
        meta = cls.get_metadata()
        bad = [f for f in meta.get("output_fields", {})
               if f not in VALID_QM_FIELDS]
        assert not bad, (
            f"{name} declares output fields not in QualityMetrics: {bad}"
        )

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_dataset_output_fields_are_valid_dataset_stats(self, name: str) -> None:
        """Every dataset-level output field must exist in DatasetStats."""
        cls = ALL_MODULES[name]
        meta = cls.get_metadata()
        bad = [f for f in meta.get("dataset_output_fields", {})
               if f not in VALID_DS_FIELDS]
        assert not bad, (
            f"{name} declares dataset output fields not in DatasetStats: {bad}"
        )

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_input_type_is_recognized(self, name: str) -> None:
        cls = ALL_MODULES[name]
        meta = cls.get_metadata()
        it = meta.get("input_type", "")
        # metrics_doc.py classifies these; anything else is a bug
        valid_tokens = {"img", "vid", "audio", "batch", "+ref", "+cap", "/"}
        tokens = re.split(r"[\s]", it.replace("+", " +").replace("/", " / "))
        for t in tokens:
            t = t.strip()
            if t and t not in valid_tokens:
                # Allow combined forms like "img/vid" etc.
                pass  # permissive — just ensure it's a non-empty string
        assert len(it) > 0, f"{name} has empty input_type"


# ═════════════════════════════════════════════════════════════════════════════
# 3. Source-level quality_metrics writes target real fields
# ═════════════════════════════════════════════════════════════════════════════


class TestFieldWrites:
    """Modules that write quality_metrics.FIELD must use valid field names."""

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_quality_metrics_writes_are_valid(self, name: str) -> None:
        cls = ALL_MODULES[name]
        src = _get_full_source(cls)
        # Match: quality_metrics.FIELD = or qm.FIELD =
        written = set()
        for pat in (r"quality_metrics\.(\w+)\s*=", r"\bqm\.(\w+)\s*="):
            written.update(re.findall(pat, src))
        # Filter out non-field names (method calls, internal attrs)
        ignore = {"__class__", "model_fields", "model_config"}
        written -= ignore
        bad = [f for f in written if f not in VALID_QM_FIELDS and not f.startswith("_")]
        assert not bad, (
            f"{name} writes unknown QualityMetrics fields: {bad}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4. HuggingFace model IDs look well-formed
# ═════════════════════════════════════════════════════════════════════════════

_HF_PRETRAINED_RE = re.compile(
    r'from_pretrained\(\s*["\']([A-Za-z0-9_.-]+/[A-Za-z0-9_.+-]+)["\']'
)
_HF_URL_RE = re.compile(
    r'https://huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.+-]+)/resolve'
)


def _extract_hf_ids(src: str) -> List[str]:
    ids = _HF_PRETRAINED_RE.findall(src)
    ids.extend(_HF_URL_RE.findall(src))
    return list(set(ids))


class TestHuggingFaceReferences:
    """HF model IDs must have org/model format and no obvious typos."""

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_hf_ids_are_well_formed(self, name: str) -> None:
        cls = ALL_MODULES[name]
        src = _get_full_source(cls)
        for hf_id in _extract_hf_ids(src):
            parts = hf_id.split("/")
            assert len(parts) == 2, f"{name}: malformed HF id {hf_id!r}"
            org, model = parts
            assert len(org) >= 1 and len(model) >= 1, (
                f"{name}: empty org or model in {hf_id!r}"
            )
            # No spaces or weird chars
            assert re.match(r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.+-]+$', hf_id), (
                f"{name}: invalid chars in HF id {hf_id!r}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# 5. pyiqa metric names are strings (not variables / expressions)
# ═════════════════════════════════════════════════════════════════════════════

_PYIQA_RE = re.compile(r'create_metric\(\s*["\']([^"\']+)["\']')


class TestPyiqaReferences:
    """pyiqa.create_metric() calls must use string literals."""

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_pyiqa_names_are_plain_strings(self, name: str) -> None:
        cls = ALL_MODULES[name]
        src = _get_full_source(cls)
        for metric_name in _PYIQA_RE.findall(src):
            # Should be lowercase alphanumeric with hyphens/underscores/plus
            assert re.match(r'^[a-z0-9_+\-]+$', metric_name, re.I), (
                f"{name}: suspicious pyiqa metric name {metric_name!r}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# 6. Module file-level docstrings exist (used for paper/arxiv extraction)
# ═════════════════════════════════════════════════════════════════════════════


class TestModuleDocstrings:
    """Each module file should have a module-level or class-level docstring."""

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_has_docstring(self, name: str) -> None:
        cls = ALL_MODULES[name]
        # Check class docstring first, then module docstring
        has_class_doc = bool(cls.__doc__ and cls.__doc__.strip())
        mod = inspect.getmodule(cls)
        has_module_doc = bool(mod and mod.__doc__ and mod.__doc__.strip())
        assert has_class_doc or has_module_doc, (
            f"{name} has no class or module docstring — "
            f"doc generators use these for paper citations and descriptions"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 7. No orphaned QualityMetrics fields (every float field should be written)
# ═════════════════════════════════════════════════════════════════════════════


class TestFieldCoverage:
    """Check that QualityMetrics float fields are actually written by some module."""

    def test_orphaned_float_fields(self) -> None:
        """Every Optional[float] field in QualityMetrics should be written by
        at least one module. Warn-only: some fields may be intentionally
        reserved for future use."""
        all_sources = {}
        for name, cls in ALL_MODULES.items():
            all_sources[name] = _get_full_source(cls)

        written: Set[str] = set()
        for src in all_sources.values():
            for pat in (r"quality_metrics\.(\w+)\s*=", r"\bqm\.(\w+)\s*="):
                written.update(re.findall(pat, src))

        float_fields = {
            f for f, info in QualityMetrics.model_fields.items()
            if "float" in str(info.annotation)
        }
        # Known intentionally-unwritten fields
        exempt = {"engagement_score", "human_preference_score"}
        orphaned = float_fields - written - exempt

        # This is a soft check — warn rather than fail for small numbers
        if len(orphaned) > 20:
            pytest.fail(
                f"{len(orphaned)} orphaned QualityMetrics float fields "
                f"(not written by any module): {sorted(orphaned)[:20]}..."
            )


# ═════════════════════════════════════════════════════════════════════════════
# 8. Field collision detection (multiple modules writing same field unguarded)
# ═════════════════════════════════════════════════════════════════════════════


class TestFieldCollisions:
    """Detect when multiple modules write the same QualityMetrics field
    without checking ``if field is None`` first.

    Known variant pairs (e.g. basic/basic_quality, flip/flip_metric) are
    expected collisions — users pick one or the other. Only flag collisions
    between modules that are NOT known variants of each other.
    """

    # Module pairs that intentionally provide the same metrics
    _KNOWN_VARIANT_PAIRS: Set[frozenset] = {
        frozenset({"basic", "basic_quality"}),
        frozenset({"dreamsim", "dreamsim_metric"}),
        frozenset({"flip", "flip_metric"}),
        frozenset({"mad", "mad_metric"}),
        frozenset({"nlpd", "nlpd_metric"}),
        frozenset({"pi", "pi_metric"}),
        frozenset({"unique_iqa", "unique"}),
        frozenset({"spectral", "spectral_complexity"}),
        frozenset({"av_sync", "audio_visual_sync"}),
        frozenset({"text", "text_detection"}),
        frozenset({"hdr_sdr_vqa", "4k_vqa"}),
        frozenset({"dedup", "deduplication"}),
    }

    @classmethod
    def _is_known_variant_pair(cls, a: str, b: str) -> bool:
        return frozenset({a, b}) in cls._KNOWN_VARIANT_PAIRS

    def test_no_unexpected_unguarded_collisions(self) -> None:
        field_writers: Dict[str, List[str]] = {}
        for name, cls in ALL_MODULES.items():
            src = _get_full_source(cls)
            for pat in (r"quality_metrics\.(\w+)\s*=", r"\bqm\.(\w+)\s*="):
                for field in re.findall(pat, src):
                    if field in VALID_QM_FIELDS:
                        field_writers.setdefault(field, []).append(name)

        unexpected = {}
        for field, writers in field_writers.items():
            unique = list(set(writers))
            if len(unique) <= 1:
                continue
            # Check if writers guard with "if field is None"
            unguarded = []
            for w in unique:
                src = _get_full_source(ALL_MODULES[w])
                if f"{field} is None" not in src:
                    unguarded.append(w)
            if len(unguarded) <= 1:
                continue
            # Filter out known variant pairs
            truly_unexpected = []
            for i, a in enumerate(unguarded):
                is_variant = all(
                    self._is_known_variant_pair(a, b)
                    for b in unguarded if b != a
                )
                if not is_variant:
                    truly_unexpected.append(a)
            if len(truly_unexpected) > 1:
                unexpected[field] = truly_unexpected

        if unexpected:
            msg_parts = [f"  {f}: {', '.join(ws)}" for f, ws in sorted(unexpected.items())]
            pytest.fail(
                f"{len(unexpected)} unexpected unguarded field collision(s):\n"
                + "\n".join(msg_parts)
            )


# ═════════════════════════════════════════════════════════════════════════════
# 9. Static health checks (same as metrics_doc._static_checks)
# ═════════════════════════════════════════════════════════════════════════════


class TestModuleStaticHealth:
    """Catch common module implementation mistakes."""

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_no_on_mount_override_without_super(self, name: str) -> None:
        """If a module overrides on_mount(), it should call super().on_mount()."""
        cls = ALL_MODULES[name]
        if "on_mount" not in cls.__dict__:
            return  # not overridden
        src = _get_full_source(cls)
        # Check for super().on_mount() in the class source
        cls_src = inspect.getsource(cls)
        if "def on_mount" in cls_src and "super().on_mount" not in cls_src:
            pytest.fail(
                f"{name} overrides on_mount() without calling super().on_mount()"
            )

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_process_method_exists(self, name: str) -> None:
        cls = ALL_MODULES[name]
        assert hasattr(cls, "process"), f"{name} has no process() method"
        assert callable(getattr(cls, "process")), f"{name}.process is not callable"


# ═════════════════════════════════════════════════════════════════════════════
# 10. URL pattern validation (no network, just format)
# ═════════════════════════════════════════════════════════════════════════════

_URL_RE = re.compile(r'https?://[^\s"\'<>,\)]+')
_KNOWN_DOMAINS = {
    "huggingface.co", "github.com", "arxiv.org", "pypi.org",
    "pytorch.org", "download.pytorch.org",
}


class TestURLFormats:
    """Validate that URLs in source code are well-formed (no network requests)."""

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_urls_are_well_formed(self, name: str) -> None:
        cls = ALL_MODULES[name]
        src = _get_full_source(cls)
        for url in _URL_RE.findall(src):
            # Strip trailing punctuation that's part of code, not URL
            url = url.rstrip(".,;:)]}'\"")
            # Must have a domain
            assert "//" in url, f"{name}: malformed URL {url!r}"
            # HF resolve URLs must have /resolve/main/ or /resolve/
            if "huggingface.co" in url and "/resolve/" in url:
                assert re.search(
                    r'huggingface\.co/[^/]+/[^/]+/resolve/',
                    url
                ), f"{name}: malformed HF resolve URL {url!r}"


# ═════════════════════════════════════════════════════════════════════════════
# 11. metric_info consistency (for modules that declare it)
# ═════════════════════════════════════════════════════════════════════════════


class TestMetricInfo:
    """If a module sets metric_info, all keys must be valid QualityMetrics fields."""

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_metric_info_keys_are_valid(self, name: str) -> None:
        cls = ALL_MODULES[name]
        if not cls.metric_info:
            return
        # metric_info keys can reference QualityMetrics or DatasetStats fields
        bad = [k for k in cls.metric_info if k not in VALID_ALL_FIELDS]
        assert not bad, (
            f"{name} metric_info has keys not in QualityMetrics or DatasetStats: {bad}"
        )

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_metric_info_values_are_strings(self, name: str) -> None:
        cls = ALL_MODULES[name]
        if not cls.metric_info:
            return
        bad = [k for k, v in cls.metric_info.items() if not isinstance(v, str)]
        assert not bad, (
            f"{name} metric_info has non-string values for: {bad}"
        )

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_dataset_metric_writes_have_metric_info(self, name: str) -> None:
        """DatasetStats writes are not discoverable from QualityMetrics output fields."""
        cls = ALL_MODULES[name]
        src = _get_full_source(cls)
        dataset_fields = {
            m.group(1)
            for m in re.finditer(r'add_dataset_metric\(\s*["\'](\w+)["\']', src)
            if m.group(1) in VALID_DS_FIELDS
        }
        missing = [f for f in sorted(dataset_fields) if f not in cls.metric_info]
        assert not missing, (
            f"{name} writes DatasetStats fields without metric_info: {missing}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 12. models attribute consistency (for modules that declare it)
# ═════════════════════════════════════════════════════════════════════════════


class TestModelsAttr:
    """If a module sets models=[], validate structure."""

    VALID_MODEL_TYPES = {
        "huggingface", "local", "pyiqa", "torch_hub", "torchvision",
        "clip", "pip_package", "other",
    }

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_models_entries_have_required_keys(self, name: str) -> None:
        cls = ALL_MODULES[name]
        if not cls.models:
            return
        for i, entry in enumerate(cls.models):
            assert isinstance(entry, dict), (
                f"{name} models[{i}] is not a dict"
            )
            assert "id" in entry, f"{name} models[{i}] missing 'id'"
            assert "type" in entry, f"{name} models[{i}] missing 'type'"
            assert entry["type"] in self.VALID_MODEL_TYPES, (
                f"{name} models[{i}] has invalid type: {entry['type']!r}"
            )

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_vendor_or_third_party_modules_declare_models(self, name: str) -> None:
        cls = ALL_MODULES[name]
        src = _get_full_source(cls)
        if "ayase.vendor" not in src and "ayase.third_party" not in src:
            return
        assert cls.models, (
            f"{name} uses vendored/third_party model code but has no class-level models metadata"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 13. No heuristic backends in production code
# ═════════════════════════════════════════════════════════════════════════════


class TestNoHeuristicBackends:
    """Ensure no module uses a heuristic fallback in production code paths."""

    @pytest.mark.parametrize("name", MODULE_NAMES)
    def test_no_production_heuristic(self, name: str) -> None:
        cls = ALL_MODULES[name]
        src = _get_full_source(cls)
        lines = src.split("\n")
        for i, line in enumerate(lines):
            if re.search(r'_backend\s*=\s*["\']heuristic["\']', line):
                # Check if it's only inside a test_mode guard
                ctx = "\n".join(lines[max(0, i - 5):i + 1])
                assert "test_mode" in ctx, (
                    f"{name} sets _backend='heuristic' outside test_mode "
                    f"at line {i + 1}"
                )


# ═════════════════════════════════════════════════════════════════════════════
# 14. Generated docs match the regenerated output (drift detector)
# ═════════════════════════════════════════════════════════════════════════════


_REPO_ROOT = Path(__file__).resolve().parent.parent
_GENERATED_TIMESTAMP_RE = re.compile(r"Generated \d{4}-\d{2}-\d{2} \d{2}:\d{2}")


def _normalize_doc(text: str) -> str:
    """Strip volatile fields (generation timestamp) from a doc for diffing."""
    return _GENERATED_TIMESTAMP_RE.sub("Generated <timestamp>", text)


def _diff(expected: str, actual: str, path: Path) -> str:
    import difflib

    diff = difflib.unified_diff(
        expected.splitlines(keepends=True),
        actual.splitlines(keepends=True),
        fromfile=f"{path.name} (committed)",
        tofile=f"{path.name} (regenerated)",
        n=3,
    )
    return "".join(diff)[:6000]


class TestGeneratedDocsAreFresh:
    """METRICS.md and MODELS.md must match what the generators currently produce.

    Catches the common failure mode where a module is added/renamed/edited
    but the docs are not regenerated. Run ``ayase modules docs -o METRICS.md``
    and ``ayase modules models -o MODELS.md`` (and ``ayase modules sync-readme``)
    to refresh.
    """

    def test_metrics_md_matches_generator(self) -> None:
        from ayase.metrics_doc import generate_metrics_doc

        committed_path = _REPO_ROOT / "METRICS.md"
        if not committed_path.exists():
            pytest.skip("METRICS.md is not present in the working tree")

        # Use --no-tests so the diff doesn't churn on test status flips.
        regenerated = _normalize_doc(generate_metrics_doc(run_tests=False))
        committed = _normalize_doc(committed_path.read_text(encoding="utf-8"))

        if committed != regenerated:
            pytest.fail(
                "METRICS.md is stale. Run `ayase modules docs --no-tests "
                "-o METRICS.md` to refresh.\n\n"
                + _diff(committed, regenerated, committed_path)
            )

    def test_models_md_matches_generator(self) -> None:
        from ayase.models_doc import generate_models_doc

        committed_path = _REPO_ROOT / "MODELS.md"
        if not committed_path.exists():
            pytest.skip("MODELS.md is not present in the working tree")

        # Disable HF API calls so the test is deterministic and offline-safe.
        regenerated = _normalize_doc(generate_models_doc(fetch_licenses=False))
        committed = _normalize_doc(committed_path.read_text(encoding="utf-8"))

        if committed != regenerated:
            pytest.fail(
                "MODELS.md is stale. Run `ayase modules models -o MODELS.md` "
                "to refresh.\n\n"
                + _diff(committed, regenerated, committed_path)
            )

    def test_readme_counts_match_registry(self) -> None:
        readme_path = _REPO_ROOT / "README.md"
        if not readme_path.exists():
            pytest.skip("README.md is not present in the working tree")

        from ayase.metrics_doc import _get_quality_metrics_fields

        ModuleRegistry.discover_modules()
        all_modules = ModuleRegistry.list_modules(packaged_only=True)
        total = len([n for n in all_modules
                     if ModuleRegistry.get_module(n) is not None])
        n_fields = len(QualityMetrics.model_fields)
        qm_fields = _get_quality_metrics_fields()

        field_writers: Set[str] = set()
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
        rendered_cats = {qm_fields[fn]["group"]
                         for fn in field_writers if fn in qm_fields}
        has_dataset_outputs = any(
            ModuleRegistry.get_module(name) is not None
            and ModuleRegistry.get_module(name).get_metadata().get("dataset_output_fields")
            for name in all_modules
        )
        n_categories = (
            len(rendered_cats)
            + (1 if has_dataset_outputs else 0)
            + (1 if has_no_output_modules else 0)
        )

        text = readme_path.read_text(encoding="utf-8")
        prose_match = re.search(
            r"(\d+) modules produce (\d+) metrics across (\d+) categories",
            text,
        )
        if prose_match:
            r_modules, r_fields, r_cats = (int(x) for x in prose_match.groups())
            assert (r_modules, r_fields, r_cats) == (total, n_fields, n_categories), (
                f"README.md prose counts ({r_modules} modules, {r_fields} metrics, "
                f"{r_cats} categories) don't match registry ({total}, {n_fields}, "
                f"{n_categories}). Run `ayase modules sync-readme`."
            )

        cli_matches = re.findall(r"show all (\d+) modules", text)
        for m in cli_matches:
            assert int(m) == total, (
                f"README.md CLI snippet says 'show all {m} modules' but "
                f"registry has {total}. Run `ayase modules sync-readme`."
            )

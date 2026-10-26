"""Pure AVGen-Bench protocol definitions, canonical metadata, and aggregation.

This module intentionally has no evaluator, model, network, or provisioning
dependencies. It implements the full-suite launcher profile from the pinned
Microsoft AVGen-Bench source and declares Ayase's stricter evidence handling.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

SOURCE_REPOSITORY = "https://github.com/microsoft/AVGen-Bench"
SOURCE_REVISION = "1049eabac472d479fe5feeb1ee202961f8e0982a"
SOURCE_SCHEME = "Scheme-2"
LAUNCHER_PROFILE = "run_full_evaluation.sh:full-suite"
SOURCE_PROTOCOL_IDENTITY = (
    f"microsoft/AVGen-Bench@{SOURCE_REVISION}/{LAUNCHER_PROFILE}/{SOURCE_SCHEME}"
)
PROTOCOL_IDENTITY = f"{SOURCE_PROTOCOL_IDENTITY}+ayase-strict-evidence-v1"

SYNCNET_LAUNCHER_CONFIDENCE_THRESHOLD = 1.0
AV_SYNC_ZERO_SCORE_THRESHOLD_SECONDS = 0.5
LIP_SYNC_ZERO_SCORE_THRESHOLD_FRAMES = 8.0
SPEECH_BACKEND = "faster-whisper"
SPEECH_MODEL = "Systran/faster-whisper-large-v3"

PROTOCOL_ADAPTATIONS = (
    "Only finite int/float observations are evidence; bool and non-finite values are rejected.",
    "A dimension with zero eligible observations is absent even if its source summary reports zero.",
    "OCR observations excluded by the released summary selection never enter the strict Text mean.",
)

DIMENSION_ORDER = (
    "visual",
    "audio",
    "av_sync",
    "lip_sync",
    "text",
    "face",
    "music",
    "speech",
    "low_physics",
    "high_physics",
    "holistic",
)

SOURCE_DIMENSION_KEYS = MappingProxyType(
    {
        "visual": "Vis",
        "audio": "Aud",
        "av_sync": "AV",
        "lip_sync": "Lip",
        "text": "Text",
        "face": "Face",
        "music": "Music",
        "speech": "Speech",
        "low_physics": "Lo-Phy",
        "high_physics": "Hi-Phy",
        "holistic": "Holistic",
    }
)
INTERNAL_DIMENSION_KEYS = MappingProxyType(
    {source: internal for internal, source in SOURCE_DIMENSION_KEYS.items()}
)

GROUP_WEIGHTS = MappingProxyType({"basic": 0.2, "cross": 0.2, "fine": 0.6})
GROUP_DIMENSIONS = MappingProxyType(
    {
        "basic": DIMENSION_ORDER[:2],
        "cross": DIMENSION_ORDER[2:4],
        "fine": DIMENSION_ORDER[4:],
    }
)
GROUP_ORDER = ("basic", "cross", "fine")

SAMPLE_SCORE_FIELDS = MappingProxyType({key: f"avgen_bench_{key}_score" for key in DIMENSION_ORDER})


@dataclass(frozen=True)
class DimensionDefinition:
    """A released dimension's raw statistic and dataset normalization."""

    key: str
    source_key: str
    group: str
    raw_field: str
    raw_unit: str
    reducer: str
    normalization: str
    sample_score_field: str
    global_weight: float


def _dimension(
    key: str,
    group: str,
    raw_field: str,
    raw_unit: str,
    reducer: str,
    normalization: str,
) -> DimensionDefinition:
    return DimensionDefinition(
        key=key,
        source_key=SOURCE_DIMENSION_KEYS[key],
        group=group,
        raw_field=raw_field,
        raw_unit=raw_unit,
        reducer=reducer,
        normalization=normalization,
        sample_score_field=SAMPLE_SCORE_FIELDS[key],
        global_weight=GROUP_WEIGHTS[group] / len(GROUP_DIMENSIONS[group]),
    )


DIMENSION_DEFINITIONS = MappingProxyType(
    {
        "visual": _dimension(
            "visual", "basic", "score", "Q-Align score", "selected_sample_mean", "raw * 100"
        ),
        "audio": _dimension(
            "audio", "basic", "PQ", "Audiobox PQ", "selected_sample_mean", "raw * 10"
        ),
        "av_sync": _dimension(
            "av_sync",
            "cross",
            "abs_pred_offset_sec_cont",
            "absolute seconds",
            "selected_sample_mean",
            "100 * (1 - raw / 0.5)",
        ),
        "lip_sync": _dimension(
            "lip_sync",
            "cross",
            "abs_offset_frames",
            "absolute frames",
            "pooled_passing_track_mean",
            "100 * (1 - raw / 8)",
        ),
        "text": _dimension(
            "text",
            "fine",
            "overall_text_quality_score",
            "0-100 score",
            "selected_sample_mean",
            "identity",
        ),
        "face": _dimension(
            "face", "fine", "final_score", "0-100 score", "selected_sample_mean", "identity"
        ),
        "music": _dimension(
            "music", "fine", "overall_score", "0-100 score", "selected_sample_mean", "identity"
        ),
        "speech": _dimension(
            "speech", "fine", "judgement.score", "0-100 score", "selected_sample_mean", "identity"
        ),
        "low_physics": _dimension(
            "low_physics",
            "fine",
            "score",
            "VideoPhy2 0-5 score",
            "selected_sample_mean",
            "raw * 20",
        ),
        "high_physics": _dimension(
            "high_physics",
            "fine",
            "overall_score",
            "0-100 score",
            "selected_sample_mean",
            "identity",
        ),
        "holistic": _dimension(
            "holistic",
            "fine",
            "plot_alignment_score",
            "0-100 score",
            "selected_sample_mean",
            "identity",
        ),
    }
)

CANONICAL_CATEGORIES = (
    "ads",
    "animals",
    "asmr",
    "chemical_reaction",
    "cooking",
    "gameplays",
    "movie_trailer",
    "musical_instrument_tutorial",
    "news",
    "physical_experiment",
    "sports",
)


@dataclass(frozen=True)
class CanonicalPrompt:
    category: str
    content: str
    prompt: str
    source_path: str


@dataclass(frozen=True)
class CanonicalFaceAnnotation:
    category: str
    content: str
    prompt: str
    annotation: Mapping[str, Any]
    source_path: str

    @property
    def min_ids(self) -> int:
        return int(self.annotation["primary_faces"]["min_ids"])

    @property
    def max_ids(self) -> int:
        return int(self.annotation["primary_faces"]["max_ids"])

    def to_dict(self) -> Dict[str, Any]:
        return _deep_thaw(self.annotation)


def is_strict_finite_number(value: object) -> bool:
    """Return whether value is a finite JSON-style number, explicitly excluding bool."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _require_dimension(dimension: str) -> DimensionDefinition:
    try:
        return DIMENSION_DEFINITIONS[dimension]
    except KeyError as exc:
        raise KeyError(f"unknown AVGen-Bench dimension: {dimension!r}") from exc


def normalize_raw_mean(dimension: str, raw_mean: object) -> float:
    """Normalize one valid dataset raw mean exactly as released Scheme-2 arithmetic."""

    _require_dimension(dimension)
    if not is_strict_finite_number(raw_mean):
        raise ValueError(f"{dimension} raw mean must be a finite int or float (not bool)")
    value = float(raw_mean)
    if dimension == "visual":
        normalized = value * 100.0
    elif dimension == "audio":
        normalized = value * 10.0
    elif dimension == "av_sync":
        normalized = 100.0 * (1.0 - value / AV_SYNC_ZERO_SCORE_THRESHOLD_SECONDS)
    elif dimension == "lip_sync":
        normalized = 100.0 * (1.0 - value / LIP_SYNC_ZERO_SCORE_THRESHOLD_FRAMES)
    elif dimension == "low_physics":
        normalized = value * 20.0
    else:
        normalized = value
    return max(0.0, min(100.0, normalized))


def _validate_dimension_mapping(name: str, values: Mapping[str, object]) -> None:
    unknown = set(values) - set(DIMENSION_ORDER)
    if unknown:
        raise KeyError(f"{name} contains unknown AVGen-Bench dimensions: {sorted(unknown)!r}")


def _validate_count(name: str, dimension: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name}[{dimension!r}] must be a non-negative int (not bool)")
    return value


def aggregate_raw_means(
    raw_means: Mapping[str, Optional[float]],
    observation_counts: Mapping[str, int],
    *,
    invalid_counts: Optional[Mapping[str, int]] = None,
    source_raw_summaries: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate strict dataset raw means using released weights and normalizations.

    A numeric raw mean is usable only with a positive evidence count. Source-generated
    summaries and counters may be attached through ``source_raw_summaries`` but never
    affect the strict score. In particular, callers must put source Music zero-at-empty
    and OCR fallback diagnostics there rather than in ``raw_means``.
    """

    invalid_counts = invalid_counts or {}
    _validate_dimension_mapping("raw_means", raw_means)
    _validate_dimension_mapping("observation_counts", observation_counts)
    _validate_dimension_mapping("invalid_counts", invalid_counts)

    dimensions: Dict[str, Dict[str, Any]] = {}
    weighted_num = 0.0
    weighted_den = 0.0
    group_values: Dict[str, list] = {group: [] for group in GROUP_ORDER}

    for key in DIMENSION_ORDER:
        definition = DIMENSION_DEFINITIONS[key]
        count = _validate_count("observation_counts", key, observation_counts.get(key, 0))
        invalid_count = _validate_count("invalid_counts", key, invalid_counts.get(key, 0))
        raw = raw_means.get(key)
        normalized: Optional[float] = None

        if count:
            if not is_strict_finite_number(raw):
                raise ValueError(f"{key} has {count} observations but no valid finite raw mean")
            raw_value = float(raw)
            if key in ("av_sync", "lip_sync") and raw_value < 0.0:
                raise ValueError(f"{key} strict raw mean must be an absolute non-negative offset")
            normalized = normalize_raw_mean(key, raw_value)
            raw = raw_value
            weighted_num += definition.global_weight * normalized
            weighted_den += definition.global_weight
            group_values[definition.group].append(normalized)
        elif raw is not None:
            raise ValueError(
                f"{key} has a raw mean but zero observations; retain it in source_raw_summaries"
            )

        dimensions[key] = {
            "source_key": definition.source_key,
            "group": definition.group,
            "raw_mean": raw,
            "normalized_score": normalized,
            "global_weight": definition.global_weight,
            "available": normalized is not None,
            "evidence_count": count,
            "invalid_observation_count": invalid_count,
            "reducer": definition.reducer,
        }

    group_scores = {
        group: (
            sum(group_values[group]) / len(group_values[group]) if group_values[group] else None
        )
        for group in GROUP_ORDER
    }
    score = weighted_num / weighted_den if weighted_den > 0.0 else None
    coverage = max(0.0, min(1.0, weighted_den))
    missing = [key for key in DIMENSION_ORDER if not dimensions[key]["available"]]

    return {
        "schema": 1,
        "protocol_identity": PROTOCOL_IDENTITY,
        "source_protocol_identity": SOURCE_PROTOCOL_IDENTITY,
        "source_revision": SOURCE_REVISION,
        "launcher_profile": LAUNCHER_PROFILE,
        "scheme": SOURCE_SCHEME,
        "declared_adaptations": list(PROTOCOL_ADAPTATIONS),
        "score": score,
        "coverage": coverage,
        "group_scores": group_scores,
        "dimensions": dimensions,
        "missing_dimensions": missing,
        "missing_source_dimensions": [SOURCE_DIMENSION_KEYS[key] for key in missing],
        "raw_source_summaries": copy.deepcopy(dict(source_raw_summaries or {})),
    }


def aggregate_observations(
    observations: Mapping[str, Iterable[object]],
    *,
    source_raw_summaries: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Pool eligible raw observations, then aggregate their dataset means.

    Pass only observations selected by the released component protocol. For Lip,
    each item must be one passing track's absolute frame offset after applying the
    launcher's inclusive confidence >= 1.0 filter; never pass per-video means.
    Invalid values are excluded and counted. Negative AV/Lip offsets are invalid
    because these observations are defined as absolute offsets.
    """

    _validate_dimension_mapping("observations", observations)
    raw_means: Dict[str, Optional[float]] = {}
    counts: Dict[str, int] = {}
    invalid_counts: Dict[str, int] = {}

    for key in DIMENSION_ORDER:
        valid = []
        invalid_count = 0
        for value in observations.get(key, ()):
            if not is_strict_finite_number(value):
                invalid_count += 1
                continue
            numeric = float(value)
            if key in ("av_sync", "lip_sync") and numeric < 0.0:
                invalid_count += 1
                continue
            valid.append(numeric)
        counts[key] = len(valid)
        invalid_counts[key] = invalid_count
        raw_means[key] = sum(valid) / len(valid) if valid else None

    return aggregate_raw_means(
        raw_means,
        counts,
        invalid_counts=invalid_counts,
        source_raw_summaries=source_raw_summaries,
    )


def safe_content_token(category: str, content: str) -> str:
    """Return a Windows-safe physical token without changing canonical identity."""

    if not isinstance(category, str) or not category:
        raise ValueError("category must be a non-empty string")
    if not isinstance(content, str) or not content:
        raise ValueError("content must be a non-empty string")
    framed_identity = f"{len(category)}:{category}{len(content)}:{content}".encode("utf-8")
    return f"avgen-{hashlib.sha256(framed_identity).hexdigest()}"


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _asset_manifest_path() -> Path:
    return _package_root() / "assets" / "manifest.json"


@lru_cache(maxsize=1)
def _asset_manifest() -> Dict[str, Any]:
    manifest = json.loads(_asset_manifest_path().read_text(encoding="utf-8"))
    if manifest.get("source_revision") != SOURCE_REVISION:
        raise ValueError("AVGen-Bench asset manifest source revision does not match protocol")
    return manifest


def asset_manifest() -> Mapping[str, Any]:
    """Return an immutable view of the generated asset digest manifest."""

    return _deep_freeze(copy.deepcopy(_asset_manifest()))


def verify_asset_bundle() -> Dict[str, Any]:
    """Verify every copied upstream asset against the generated manifest."""

    root = _package_root()
    failures = []
    checked = 0
    for entry in _asset_manifest()["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            failures.append({"path": entry["path"], "reason": "unsafe manifest path"})
            continue
        path = root / relative
        try:
            data = path.read_bytes()
        except OSError as exc:
            failures.append({"path": entry["path"], "reason": str(exc)})
            continue
        checked += 1
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if len(data) != entry["bytes"] or actual_sha256 != entry["sha256"]:
            failures.append(
                {
                    "path": entry["path"],
                    "reason": "digest or byte-length mismatch",
                    "expected_bytes": entry["bytes"],
                    "actual_bytes": len(data),
                    "expected_sha256": entry["sha256"],
                    "actual_sha256": actual_sha256,
                }
            )
    return {"valid": not failures, "checked": checked, "failures": failures}


def _manifest_entries(kind: str) -> Sequence[Mapping[str, Any]]:
    return tuple(entry for entry in _asset_manifest()["files"] if entry["kind"] == kind)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


@lru_cache(maxsize=1)
def canonical_prompts() -> Tuple[CanonicalPrompt, ...]:
    """Load the exact 235 canonical prompt records in manifest/category order."""

    records = []
    for entry in _manifest_entries("prompts"):
        raw_records = json.loads((_package_root() / entry["path"]).read_text(encoding="utf-8"))
        if len(raw_records) != entry["records"]:
            raise ValueError(f"canonical prompt count mismatch in {entry['path']}")
        for raw in raw_records:
            records.append(
                CanonicalPrompt(
                    category=entry["category"],
                    content=raw["content"],
                    prompt=raw["prompt"],
                    source_path=entry["source_path"],
                )
            )
    return tuple(records)


@lru_cache(maxsize=1)
def canonical_face_annotations() -> Tuple[CanonicalFaceAnnotation, ...]:
    """Load the exact released 220-record canonical face-annotation universe."""

    records = []
    for entry in _manifest_entries("face_annotations"):
        raw_records = json.loads((_package_root() / entry["path"]).read_text(encoding="utf-8"))
        if len(raw_records) != entry["records"]:
            raise ValueError(f"canonical face-annotation count mismatch in {entry['path']}")
        for raw in raw_records:
            records.append(
                CanonicalFaceAnnotation(
                    category=entry["category"],
                    content=raw["content"],
                    prompt=raw["prompt"],
                    annotation=_deep_freeze(raw),
                    source_path=entry["source_path"],
                )
            )
    return tuple(records)


@lru_cache(maxsize=1)
def _prompt_index() -> Mapping[Tuple[str, str], CanonicalPrompt]:
    return MappingProxyType(
        {(record.category, record.content): record for record in canonical_prompts()}
    )


@lru_cache(maxsize=1)
def _face_index() -> Mapping[Tuple[str, str], CanonicalFaceAnnotation]:
    return MappingProxyType(
        {(record.category, record.content): record for record in canonical_face_annotations()}
    )


def lookup_prompt(category: str, content: str) -> CanonicalPrompt:
    """Resolve a canonical prompt only by its exact explicit category/content identity."""

    try:
        return _prompt_index()[(category, content)]
    except KeyError as exc:
        raise KeyError(
            f"unknown canonical AVGen-Bench prompt identity: {(category, content)!r}"
        ) from exc


def lookup_face_annotation(category: str, content: str) -> CanonicalFaceAnnotation:
    """Resolve a released face label only by exact explicit category/content identity."""

    try:
        return _face_index()[(category, content)]
    except KeyError as exc:
        raise KeyError(
            f"unknown canonical AVGen-Bench face-annotation identity: {(category, content)!r}"
        ) from exc


def missing_face_annotation_identities() -> Tuple[Tuple[str, str], ...]:
    """Return the canonical prompts absent from the released 220-label face universe."""

    missing = set(_prompt_index()) - set(_face_index())
    return tuple(sorted(missing))


def canonical_data_summary() -> Dict[str, Any]:
    """Return canonical record and released face-applicability counts by category."""

    summary = {}
    prompts = canonical_prompts()
    annotations = canonical_face_annotations()
    for category in CANONICAL_CATEGORIES:
        category_prompts = [record for record in prompts if record.category == category]
        category_annotations = [record for record in annotations if record.category == category]
        summary[category] = {
            "prompts": len(category_prompts),
            "face_annotations": len(category_annotations),
            "face_max_ids_positive": sum(record.max_ids > 0 for record in category_annotations),
        }
    return {
        "source_revision": SOURCE_REVISION,
        "categories": summary,
        "total_prompts": len(prompts),
        "total_face_annotations": len(annotations),
        "face_max_ids_positive": sum(record.max_ids > 0 for record in annotations),
        "missing_face_annotations": [
            {"category": category, "content": content}
            for category, content in missing_face_annotation_identities()
        ],
    }

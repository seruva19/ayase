"""Complete AVGen-Bench full-suite evaluation over eleven released dimensions.

Scores are produced dataset-major in ``post_process`` so each isolated specialist
loads once, evaluates every selected MP4, then exits before the next specialist.
All per-sample scores are normalized to 0-100 (higher is better). The official
suite total remains dataset-only and uses raw dataset means before normalization.

Canonical mode requires exact ``category``/``content`` identities and uses the
bundled 235 prompts plus the released 220-record face-label universe. Custom mode
uses ``sample.caption.text`` first, an explicit prompt only as fallback, and keeps
promptless MP4s eligible for prompt-independent native heads. It uses the literal
``custom`` category by default and stable order-based content identities.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.third_party.avgen_bench.protocol import (
    DIMENSION_ORDER,
    LAUNCHER_PROFILE,
    PROTOCOL_ADAPTATIONS,
    PROTOCOL_IDENTITY,
    SAMPLE_SCORE_FIELDS,
    SOURCE_DIMENSION_KEYS,
    SOURCE_PROTOCOL_IDENTITY,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    SYNCNET_LAUNCHER_CONFIDENCE_THRESHOLD,
    aggregate_observations,
    canonical_data_summary,
    canonical_prompts,
    is_strict_finite_number,
    lookup_face_annotation,
    lookup_prompt,
    missing_face_annotation_identities,
    normalize_raw_mean,
)

logger = logging.getLogger(__name__)

_API_DIMENSIONS = frozenset({"text", "music", "speech", "high_physics", "holistic"})
_FFMPEG_DIMENSIONS = frozenset({"av_sync", "lip_sync", "music"})
_BAD_COMPONENT_STATUSES = frozenset(
    {"failed", "missing_credentials", "missing_prompt", "missing_video"}
)
_CATEGORY_PATTERN = re.compile(r"[a-z0-9_-]+")
_RESERVED_CATEGORIES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
_SECRET_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|password|secret|credential|access[_-]?token|auth[_-]?token)", re.I
)
_WORKER_CONFIG_KEYS = frozenset(
    {
        "batch_size",
        "compute_type",
        "crop_scale",
        "ctx_id",
        "dbscan_eps",
        "dbscan_min_samples",
        "det_size",
        "device",
        "facedet_scale",
        "frame_rate",
        "gemini_model",
        "gemini_retries",
        "max_gap_frames",
        "min_cos",
        "min_det_score",
        "min_face_px",
        "min_face_size",
        "min_iou",
        "min_primary_cluster_dets",
        "min_rel_area",
        "min_track",
        "min_tracklet_len",
        "no_reencode",
        "num_failed_det",
        "primary_ratio_thresh",
        "sample_rate",
        "timeout_s",
        "v_start_i_sec",
        "vshift",
        "workers",
    }
)

_SAMPLE_METRIC_INFO = {
    "visual": "Q-Align joint-video visual quality (0-100, higher=better)",
    "audio": "Audiobox production quality (0-100, higher=better)",
    "av_sync": "Synchformer audiovisual synchronization quality (0-100, higher=better)",
    "lip_sync": "SyncNet passing-track lip synchronization quality (0-100, higher=better)",
    "text": "OCR and visible-text fidelity (0-100, higher=better)",
    "face": "Expected-face count and identity consistency (0-100, higher=better)",
    "music": "Prompt-conditioned musical correctness (0-100, higher=better)",
    "speech": "Prompt-conditioned speech compliance (0-100, higher=better)",
    "low_physics": "VideoPhy-2 low-level physical plausibility (0-100, higher=better)",
    "high_physics": "Gemini high-level physical plausibility (0-100, higher=better)",
    "holistic": "Holistic audiovisual plot alignment (0-100, higher=better)",
}

_DATASET_METRIC_INFO = {
    "avgen_bench_score": "AVGen-Bench Scheme-2 dataset score (0-100, higher=better)",
    "avgen_bench_basic_score": "Mean available visual/audio component score (0-100)",
    "avgen_bench_cross_score": "Mean available AV/lip synchronization score (0-100)",
    "avgen_bench_fine_score": "Mean available fine-grained component score (0-100)",
    "avgen_bench_coverage": "Available fixed AVGen-Bench dimension weight (0-1)",
    "avgen_bench_metrics": "Structured component evidence, counts, and protocol identities",
}


def _load_runtime() -> Any:
    from ayase.third_party.avgen_bench import runtime

    return runtime


def _load_assets_runtime() -> Any:
    from ayase.third_party.avgen_bench import assets_runtime

    return assets_runtime


def _bounded_error(exc: BaseException, secrets: Sequence[str] = ()) -> str:
    text = f"{type(exc).__name__}: {exc}"
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text[:1200]


def _json_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _json_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_paths(item) for item in value]
    return value


def _clean_nonsecret_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _clean_nonsecret_value(item)
            for key, item in value.items()
            if not _SECRET_KEY_PATTERN.search(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_clean_nonsecret_value(item) for item in value]
    return _json_paths(value)


def _clean_nonsecret_mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    clean: Dict[str, Any] = {}
    for key, item in value.items():
        name = str(key)
        if _SECRET_KEY_PATTERN.search(name):
            continue
        clean[name] = _clean_nonsecret_value(item)
    return clean


class AVGenBenchModule(PipelineModule):
    """Microsoft AVGen-Bench full-suite orchestration with strict evidence handling."""

    name = "avgen_bench"
    implementation_status = "implemented"
    implementation_notes = "Eleven-dimension execution bridge stages records, provisions pinned source/runtime/assets and calls specialist workers; published normalization and Scheme-2 aggregation retained, absent evidence is explicit. Static gap: all specialist dependency trees and model execution were not exhaustively reproduced."
    implementation_review = "Source inspection; confidence medium; pretrained inference not validated."
    description = "Complete eleven-dimension Microsoft AVGen-Bench full-suite evaluation"
    input_type = "video"
    required_packages: List[str] = []
    metric_info = {
        **{SAMPLE_SCORE_FIELDS[key]: description for key, description in _SAMPLE_METRIC_INFO.items()},
        **_DATASET_METRIC_INFO,
    }
    metric_groups = {
        SAMPLE_SCORE_FIELDS["visual"]: "nr_quality",
        SAMPLE_SCORE_FIELDS["audio"]: "audio",
        SAMPLE_SCORE_FIELDS["av_sync"]: "audio",
        SAMPLE_SCORE_FIELDS["lip_sync"]: "audio",
        SAMPLE_SCORE_FIELDS["text"]: "alignment",
        SAMPLE_SCORE_FIELDS["face"]: "face",
        SAMPLE_SCORE_FIELDS["music"]: "audio",
        SAMPLE_SCORE_FIELDS["speech"]: "audio",
        SAMPLE_SCORE_FIELDS["low_physics"]: "alignment",
        SAMPLE_SCORE_FIELDS["high_physics"]: "alignment",
        SAMPLE_SCORE_FIELDS["holistic"]: "alignment",
    }
    provenance = {
        **{field: "adapted" for field in SAMPLE_SCORE_FIELDS.values()},
        "avgen_bench_score": "adapted",
        "avgen_bench_basic_score": "adapted",
        "avgen_bench_cross_score": "adapted",
        "avgen_bench_fine_score": "adapted",
        "avgen_bench_coverage": "adapted",
        "avgen_bench_metrics": "utility",
    }
    sources = {
        field: (
            "Microsoft AVGen-Bench full-suite source at "
            f"{SOURCE_REPOSITORY}/tree/{SOURCE_REVISION}"
        )
        for field in (
            *SAMPLE_SCORE_FIELDS.values(),
            "avgen_bench_score",
            "avgen_bench_basic_score",
            "avgen_bench_cross_score",
            "avgen_bench_fine_score",
            "avgen_bench_coverage",
        )
    } | {"avgen_bench_metrics": "Ayase AVGen-Bench runtime and evidence ledger"}
    deviations = {
        field: (
            "Runs the pinned released full-suite component and Scheme-2 arithmetic, while Ayase "
            "validates judge JSON before numeric coercion, rejects boolean/non-finite raw values, "
            "and treats zero selected OCR/Music evidence as absent instead of fabricating a "
            "numeric dimension. Promptless custom MP4s retain prompt-independent native heads; "
            "judge rows without prompts, Face rows without positive labels, and canonical "
            "Music rows outside the selected category are preflighted before specialist "
            "provisioning. Missing dimensions are "
            "explicitly renormalized and reported through coverage; repaired edge cases do not "
            "claim historical leaderboard parity."
        )
        for field in (
            *SAMPLE_SCORE_FIELDS.values(),
            "avgen_bench_score",
            "avgen_bench_basic_score",
            "avgen_bench_cross_score",
            "avgen_bench_fine_score",
            "avgen_bench_coverage",
        )
    } | {"avgen_bench_metrics": "Ayase-owned structured execution and evidence metadata."}
    implementation_sources = (
        "../third_party/avgen_bench/LICENSE",
        "../third_party/avgen_bench/NOTICE",
        "../third_party/avgen_bench/protocol.py",
        "../third_party/avgen_bench/runtime.py",
        "../third_party/avgen_bench/assets_runtime.py",
        "../third_party/avgen_bench/_native_runner.py",
        "../third_party/avgen_bench/adapters_native.py",
        "../third_party/avgen_bench/adapters_remote.py",
    )
    models = [
        {
            "id": "q-future/one-align",
            "type": "huggingface",
            "revision": "dcc603b95aa0ebd82afa696d4a1e20d11fc80ddb",
            "task": "AVGen visual quality (Q-Align joint-video scoring)",
            "auto_download": True,
            "notes": "Evaluator carries S-Lab noncommercial terms; checkpoint has a Llama-2 chain.",
        },
        {
            "id": "facebook/audiobox-aesthetics",
            "type": "huggingface",
            "revision": "9b1dd8e5df9af7216e836a98974fe3b82c56ded6",
            "url": "https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt",
            "task": "AVGen audio production quality",
            "auto_download": True,
        },
        {
            "id": "Synchformer/24-01-04T16-39-21",
            "type": "local",
            "url": "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt",
            "task": "Continuous audiovisual offset prediction",
            "auto_download": True,
        },
        {
            "id": "syncnet_v2.model",
            "type": "local",
            "url": "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
            "task": "Lip synchronization track scoring",
            "auto_download": True,
            "notes": "Oxford model page grants research use under CC BY 4.0.",
        },
        {
            "id": "sfd_face.pth",
            "type": "local",
            "url": "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth",
            "task": "SyncNet S3FD face detection",
            "auto_download": True,
            "notes": "A separate pretrained-weight grant was not established.",
        },
        {
            "id": "PaddlePaddle/PP-OCRv5_server_det",
            "type": "huggingface",
            "revision": "ca867c897ecbca8873081573a802ad70d499cb94",
            "task": "AVGen visible-text detection",
            "auto_download": True,
        },
        {
            "id": "PaddlePaddle/PP-OCRv5_server_rec",
            "type": "huggingface",
            "revision": "b26c3587fda8da3c8ec0ce357214b4d661ff1558",
            "task": "AVGen visible-text recognition",
            "auto_download": True,
        },
        {
            "id": "InsightFace/buffalo_l",
            "type": "local",
            "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
            "task": "AVGen face detection and recognition",
            "auto_download": True,
            "notes": "InsightFace pretrained model pack is restricted to noncommercial research.",
        },
        {
            "id": "spotify/basic-pitch:nmp.onnx",
            "type": "local",
            "url": "https://raw.githubusercontent.com/spotify/basic-pitch/v0.4.0/basic_pitch/saved_models/icassp_2022/nmp.onnx",
            "task": "AVGen music note and chord extraction",
            "auto_download": True,
        },
        {
            "id": "Systran/faster-whisper-large-v3",
            "type": "huggingface",
            "revision": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
            "task": "AVGen speech transcription",
            "auto_download": True,
        },
        {
            "id": "videophysics/videophy_2_auto",
            "type": "huggingface",
            "revision": "ca38370e1dcc79c99c7017dd8755d0a087299afc",
            "task": "AVGen low-level physical plausibility",
            "auto_download": True,
            "notes": "Original mPLUG/Llama-derived base terms remain separately applicable.",
        },
        {
            "id": "gemini-3-flash-preview",
            "type": "other",
            "task": "OCR, music, speech, high-physics, and holistic AVGen judgments",
            "auto_download": False,
            "notes": "Optional paid Gemini API service; credentials are never persisted in requests.",
        },
    ]
    default_config = {
        "input_mode": "custom",
        "input_mapping": {},
        "category": None,
        "content": None,
        "prompt": None,
        "expected_faces": None,
        "models_dir": "models",
        "source_root": None,
        "asset_overrides": {},
        "dimension_config": {},
        "api_key": None,
        "gemini_model": "gemini-3-flash-preview",
        "device": "auto",
        "workers": 1,
        "timeout_s": 600,
        "worker_timeout_s": 14400,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._backend: Optional[str] = None
        self._ml_available = False
        self.last_metadata: Dict[str, Any] = {}

    def setup(self) -> None:
        """Defer every source, runtime, model, and API action until post-processing."""

        if self.test_mode:
            return
        mode = self.config.get("input_mode")
        if mode not in {"canonical", "custom"}:
            logger.warning("AVGen-Bench input_mode must be 'canonical' or 'custom'")
            self._backend = "unavailable"
            return
        self._backend = "deferred:eleven-specialists"
        self._ml_available = True

    def process(self, sample: Sample) -> Sample:
        """Return the identical sample; scores become available in ``post_process``."""

        return sample

    @staticmethod
    def _reset_sample_outputs(samples: Sequence[Sample]) -> None:
        declared = QualityMetrics.model_fields
        for sample in samples:
            sample.metadata.pop("avgen_bench", None)
            if sample.quality_metrics is None:
                continue
            for field in SAMPLE_SCORE_FIELDS.values():
                if field in declared:
                    setattr(sample.quality_metrics, field, None)

    def _publish_dataset(self, aggregate: Mapping[str, Any], metadata: Dict[str, Any]) -> None:
        self.last_metadata = metadata
        if self.pipeline is None:
            return
        groups = aggregate["group_scores"]
        self.pipeline.add_dataset_metric("avgen_bench_score", aggregate["score"])
        self.pipeline.add_dataset_metric("avgen_bench_basic_score", groups["basic"])
        self.pipeline.add_dataset_metric("avgen_bench_cross_score", groups["cross"])
        self.pipeline.add_dataset_metric("avgen_bench_fine_score", groups["fine"])
        self.pipeline.add_dataset_metric("avgen_bench_coverage", aggregate["coverage"])
        self.pipeline.add_dataset_metric("avgen_bench_metrics", metadata)

    def _mapping_for_sample(
        self, sample: Sample, basename_counts: Mapping[str, int]
    ) -> Tuple[Optional[Mapping[str, Any]], Optional[str]]:
        configured = self.config.get("input_mapping", {})
        if not isinstance(configured, Mapping):
            return None, "input_mapping must be a mapping"
        direct = str(sample.path)
        try:
            resolved = str(Path(sample.path).expanduser().resolve())
        except (OSError, TypeError, ValueError):
            resolved = direct
        for key in (direct, resolved):
            if key in configured:
                row = configured[key]
                return (
                    (row, None)
                    if isinstance(row, Mapping)
                    else (None, f"input_mapping[{key!r}] must be an object")
                )
        basename = Path(sample.path).name
        if basename in configured:
            if basename_counts.get(basename, 0) != 1:
                return None, f"ambiguous basename input_mapping key: {basename!r}"
            row = configured[basename]
            return (
                (row, None)
                if isinstance(row, Mapping)
                else (None, f"input_mapping[{basename!r}] must be an object")
            )
        return None, None

    def _input_value(
        self,
        sample: Sample,
        mapping: Optional[Mapping[str, Any]],
        name: str,
    ) -> Tuple[Any, Optional[str]]:
        metadata_name = "avgen_bench_" + name
        if metadata_name in sample.metadata:
            return sample.metadata[metadata_name], "sample_metadata"
        if mapping is not None:
            if metadata_name in mapping:
                return mapping[metadata_name], "input_mapping"
            if name in mapping:
                return mapping[name], "input_mapping"
        if self.config.get(name) is not None:
            return self.config[name], "module_config"
        return None, None

    @staticmethod
    def _validate_expected_faces(value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("expected_faces must be a full annotation object")
        primary = value.get("primary_faces")
        if not isinstance(primary, Mapping):
            raise ValueError("expected_faces.primary_faces must be an object")
        lower, upper = primary.get("min_ids"), primary.get("max_ids")
        if (
            type(lower) is not int
            or type(upper) is not int
            or lower < 0
            or upper < lower
        ):
            raise ValueError("expected_faces requires integer 0 <= min_ids <= max_ids")
        return copy.deepcopy(dict(value))

    def _record_for_sample(
        self,
        sample: Sample,
        index: int,
        mapping: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        mode = str(self.config.get("input_mode", ""))
        if mode not in {"canonical", "custom"}:
            raise ValueError("input_mode must be exactly 'canonical' or 'custom'")
        if not sample.is_video:
            raise ValueError("AVGen-Bench accepts video samples only")
        path = Path(sample.path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"video is missing: {path}")
        if path.suffix.lower() != ".mp4":
            raise ValueError(
                "the released AVGen-Bench suite requires MP4 input; no lossy transcoding is applied"
            )

        category, category_source = self._input_value(sample, mapping, "category")
        content, content_source = self._input_value(sample, mapping, "content")
        supplied_prompt, supplied_prompt_source = self._input_value(sample, mapping, "prompt")
        supplied_faces, supplied_faces_source = self._input_value(
            sample, mapping, "expected_faces"
        )
        ignored_overrides: List[str] = []

        if mode == "canonical":
            if not isinstance(category, str) or not category:
                raise ValueError("canonical mode requires an explicit avgen_bench_category")
            if not isinstance(content, str) or not content:
                raise ValueError("canonical mode requires an explicit avgen_bench_content")
            canonical = lookup_prompt(category, content)
            prompt = canonical.prompt
            prompt_source = canonical.source_path
            try:
                expected_faces = lookup_face_annotation(category, content).to_dict()
                face_source = "released_face_annotation"
            except KeyError:
                expected_faces = None
                face_source = "released_annotation_missing"
            if supplied_prompt is not None:
                ignored_overrides.append(supplied_prompt_source + ":prompt")
            if supplied_faces is not None:
                ignored_overrides.append(supplied_faces_source + ":expected_faces")
        else:
            if category is None:
                category, category_source = "custom", "documented_default"
            if not isinstance(category, str) or not _CATEGORY_PATTERN.fullmatch(category):
                raise ValueError("custom category must match [a-z0-9_-]+")
            if category.lower() in _RESERVED_CATEGORIES:
                raise ValueError("custom category is a reserved Windows directory name")
            if content is None:
                content, content_source = f"custom-{index + 1:06d}", "stable_sample_order"
            if not isinstance(content, str) or not content:
                raise ValueError("custom content must be a non-empty string")
            caption = None
            if sample.caption is not None and isinstance(sample.caption.text, str):
                caption = sample.caption.text.strip() or None
            if caption is not None:
                prompt, prompt_source = caption, "sample_caption"
            elif isinstance(supplied_prompt, str) and supplied_prompt.strip():
                prompt, prompt_source = supplied_prompt.strip(), supplied_prompt_source
            else:
                prompt, prompt_source = "", "absent"
            expected_faces = self._validate_expected_faces(supplied_faces)
            face_source = supplied_faces_source if expected_faces is not None else None

        return {
            "id": f"sample-{index + 1:06d}",
            "video_path": str(path),
            "category": category,
            "content": content,
            "prompt_text": prompt,
            "expected_faces": expected_faces,
            "mode": mode,
            "_input_provenance": {
                "category": category_source,
                "content": content_source,
                "prompt": prompt_source,
                "expected_faces": face_source,
                "ignored_canonical_overrides": ignored_overrides,
            },
        }

    def _records_for_samples(
        self, samples: Sequence[Sample]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Sample], int]:
        basename_counts: Dict[str, int] = {}
        for sample in samples:
            name = Path(sample.path).name
            basename_counts[name] = basename_counts.get(name, 0) + 1

        rows: List[Tuple[Sample, Dict[str, Any]]] = []
        invalid = 0
        for index, sample in enumerate(samples):
            mapping, mapping_error = self._mapping_for_sample(sample, basename_counts)
            try:
                if mapping_error:
                    raise ValueError(mapping_error)
                record = self._record_for_sample(sample, index, mapping)
                rows.append((sample, record))
                sample.metadata["avgen_bench"] = {
                    "schema": 1,
                    "status": "pending",
                    "mode": record["mode"],
                    "source_revision": SOURCE_REVISION,
                    "launcher_profile": LAUNCHER_PROFILE,
                    "input": {
                        "category": record["category"],
                        "content": record["content"],
                        "prompt": record["prompt_text"],
                        "prompt_sha256": hashlib.sha256(
                            record["prompt_text"].encode("utf-8")
                        ).hexdigest(),
                        "expected_faces": copy.deepcopy(record["expected_faces"]),
                        "provenance": copy.deepcopy(record["_input_provenance"]),
                    },
                    "components": {},
                }
            except Exception as exc:
                invalid += 1
                sample.metadata["avgen_bench"] = {
                    "schema": 1,
                    "status": "invalid_input",
                    "mode": self.config.get("input_mode"),
                    "source_revision": SOURCE_REVISION,
                    "error": _bounded_error(exc),
                    "components": {},
                }

        identities: Dict[Tuple[str, str], List[Tuple[Sample, Dict[str, Any]]]] = {}
        for item in rows:
            record = item[1]
            identities.setdefault((record["category"], record["content"]), []).append(item)
        duplicates = {identity for identity, items in identities.items() if len(items) > 1}
        records: List[Dict[str, Any]] = []
        by_id: Dict[str, Sample] = {}
        for sample, record in rows:
            identity = (record["category"], record["content"])
            if identity in duplicates:
                invalid += 1
                sample.metadata["avgen_bench"].update(
                    status="invalid_input",
                    error=f"duplicate AVGen-Bench category/content identity: {identity!r}",
                )
                continue
            record.pop("_input_provenance", None)
            records.append(record)
            by_id[record["id"]] = sample
        return records, by_id, invalid

    def _api_key(self) -> Optional[str]:
        explicit = self.config.get("api_key")
        if isinstance(explicit, str) and explicit:
            return explicit
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None

    def _worker_config(self, internal: str) -> Dict[str, Any]:
        base = {
            key: self.config[key]
            for key in _WORKER_CONFIG_KEYS
            if key in self.config and self.config[key] is not None
        }
        configured = self.config.get("dimension_config", {})
        if isinstance(configured, Mapping):
            source_key = SOURCE_DIMENSION_KEYS[internal]
            selected = configured.get(source_key, configured.get(internal, {}))
            if isinstance(selected, Mapping):
                base.update(selected)
        clean = _clean_nonsecret_mapping(base)
        if clean.get("device") == "auto" and internal in {
            "visual",
            "av_sync",
            "low_physics",
        }:
            # These released GPU evaluators accept concrete torch devices only;
            # omitting Ayase's generic sentinel retains each source adapter's
            # exact default (cuda, cuda:0, and cuda respectively).
            clean.pop("device")
        return clean

    def _asset_overrides(
        self, internal: str, assets_runtime: Any
    ) -> Optional[Dict[str, Any]]:
        configured = self.config.get("asset_overrides", {})
        if not isinstance(configured, Mapping) or not configured:
            return None
        source_key = SOURCE_DIMENSION_KEYS[internal]
        dimension_names = set(DIMENSION_ORDER) | set(SOURCE_DIMENSION_KEYS.values())
        provided_names = {str(key) for key in configured}
        if provided_names & dimension_names:
            unknown = provided_names - dimension_names
            if unknown:
                raise ValueError(
                    f"unknown dimension keys in nested asset_overrides: {sorted(unknown)!r}"
                )
            selected = configured.get(source_key, configured.get(internal))
            if selected is None:
                return None
            if not isinstance(selected, Mapping):
                raise ValueError(f"asset_overrides[{source_key!r}] must be an object")
            return _clean_nonsecret_mapping(selected)

        all_asset_keys = {
            asset_key
            for dimension in SOURCE_DIMENSION_KEYS.values()
            for asset_key in assets_runtime.required_asset_keys(dimension)
        }
        nonsecret = _clean_nonsecret_mapping(configured)
        unknown = set(nonsecret) - all_asset_keys
        if unknown:
            raise ValueError(f"unknown flat AVGen-Bench asset override keys: {sorted(unknown)!r}")
        required = set(assets_runtime.required_asset_keys(source_key))
        selected = {key: value for key, value in nonsecret.items() if key in required}
        return selected or None

    @staticmethod
    def _missing_credentials_result(source_key: str, records: Sequence[Mapping[str, Any]]) -> dict:
        rows = [
            {
                "id": row["id"],
                "status": "missing_credentials",
                "eligible": None,
                "raw_score": None,
                "raw_details": {},
                "error": "Gemini credentials are unavailable",
            }
            for row in records
        ]
        return {
            "schema": 1,
            "source_revision": SOURCE_REVISION,
            "dimension": source_key,
            "records": rows,
            "raw_summary": {"status": "missing_credentials"},
            "counts": {"requested": len(rows), "scored": 0, "failed": 0},
        }

    @staticmethod
    def _failed_result(
        source_key: str, records: Sequence[Mapping[str, Any]], error: str
    ) -> dict:
        rows = [
            {
                "id": row["id"],
                "status": "failed",
                "eligible": None,
                "raw_score": None,
                "raw_details": {},
                "error": error,
            }
            for row in records
        ]
        return {
            "schema": 1,
            "source_revision": SOURCE_REVISION,
            "dimension": source_key,
            "records": rows,
            "raw_summary": {"error": error},
            "counts": {"requested": len(rows), "scored": 0, "failed": len(rows)},
        }

    @staticmethod
    def _result_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
        counts = {
            "requested": len(rows),
            "located": len(rows),
            "eligibility_known_true": 0,
            "eligibility_known_false": 0,
            "eligibility_unknown": 0,
            "scored": 0,
            "failed": 0,
            "ineligible": 0,
            "missing_prompt": 0,
            "missing_video": 0,
            "missing_face_label": 0,
            "no_face_detection": 0,
        }
        for row in rows:
            eligible = row.get("eligible")
            if eligible is True:
                counts["eligibility_known_true"] += 1
            elif eligible is False:
                counts["eligibility_known_false"] += 1
            else:
                counts["eligibility_unknown"] += 1
            status = row.get("status")
            if isinstance(status, str) and status in counts:
                counts[status] += 1
        counts["summary_eligible_numeric_count"] = counts["scored"]
        return counts

    @staticmethod
    def _preflight_row(
        record: Mapping[str, Any],
        status: str,
        eligible: Optional[bool],
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "id": record["id"],
            "status": status,
            "eligible": eligible,
            "raw_score": None,
            "raw_details": copy.deepcopy(dict(details or {})),
        }

    @classmethod
    def _preflight_result(
        cls,
        source_key: str,
        records: Sequence[Mapping[str, Any]],
        rows_by_id: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        rows = [copy.deepcopy(dict(rows_by_id[row["id"]])) for row in records]
        return {
            "schema": 1,
            "source_revision": SOURCE_REVISION,
            "dimension": source_key,
            "records": rows,
            "raw_summary": {
                "count_scored": 0,
                "raw_sum": 0.0,
                "raw_mean": None,
                "preflight_only": True,
            },
            "counts": cls._result_counts(rows),
        }

    @classmethod
    def _merge_preflight_rows(
        cls,
        source_key: str,
        records: Sequence[Mapping[str, Any]],
        result: Mapping[str, Any],
        preflight_rows: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if not preflight_rows:
            return copy.deepcopy(dict(result))
        worker_rows = result.get("records", [])
        by_id = {
            row.get("id"): row
            for row in worker_rows
            if isinstance(row, Mapping) and isinstance(row.get("id"), str)
        }
        by_id.update(preflight_rows)
        rows = [copy.deepcopy(dict(by_id[row["id"]])) for row in records]
        merged = copy.deepcopy(dict(result))
        merged.update(
            schema=1,
            source_revision=SOURCE_REVISION,
            dimension=source_key,
            records=rows,
            counts=cls._result_counts(rows),
        )
        raw_summary = merged.get("raw_summary")
        if not isinstance(raw_summary, dict):
            raw_summary = {}
            merged["raw_summary"] = raw_summary
        raw_summary["public_preflight"] = {
            status: sum(row.get("status") == status for row in preflight_rows.values())
            for status in sorted({str(row.get("status")) for row in preflight_rows.values()})
        }
        return merged

    @classmethod
    def _dimension_plan(
        cls,
        internal: str,
        records: Sequence[Mapping[str, Any]],
    ) -> Tuple[List[Mapping[str, Any]], Dict[str, Dict[str, Any]]]:
        if internal == "face":
            has_positive_label = any(
                isinstance(row.get("expected_faces"), Mapping)
                and isinstance(row["expected_faces"].get("primary_faces"), Mapping)
                and type(row["expected_faces"]["primary_faces"].get("max_ids")) is int
                and row["expected_faces"]["primary_faces"]["max_ids"] > 0
                for row in records
            )
            if has_positive_label:
                # The released wrapper produces all mixed positive/missing/zero-label rows.
                return list(records), {}
            preflight: Dict[str, Dict[str, Any]] = {}
            for row in records:
                annotation = row.get("expected_faces")
                primary = annotation.get("primary_faces") if isinstance(annotation, Mapping) else None
                if isinstance(primary, Mapping) and primary.get("max_ids") == 0:
                    preflight[row["id"]] = cls._preflight_row(
                        row,
                        "ineligible",
                        False,
                        details={
                            "expected_primary_faces": {
                                "min_ids": primary.get("min_ids"),
                                "max_ids": 0,
                            },
                            "reason": "source skips labels with max_ids=0",
                        },
                    )
                else:
                    preflight[row["id"]] = cls._preflight_row(
                        row, "missing_face_label", None
                    )
            return [], preflight

        if internal == "music":
            applicable = [
                row
                for row in records
                if row.get("mode") != "canonical"
                or row.get("category") == "musical_instrument_tutorial"
            ]
            preflight = {
                row["id"]: cls._preflight_row(
                    row,
                    "ineligible",
                    False,
                    details={
                        "source_selection": (
                            "canonical launcher selects musical_instrument_tutorial only"
                        )
                    },
                )
                for row in records
            }
            if not applicable:
                return [], preflight

        if internal in _API_DIMENSIONS:
            runnable = []
            preflight = {}
            for row in records:
                prompt = row.get("prompt_text")
                if isinstance(prompt, str) and prompt.strip():
                    runnable.append(row)
                else:
                    preflight[row["id"]] = cls._preflight_row(
                        row, "missing_prompt", None
                    )
            return runnable, preflight
        return list(records), {}

    @staticmethod
    def _normalized_row_score(internal: str, row: Mapping[str, Any]) -> Optional[float]:
        raw = row.get("raw_score")
        if row.get("status") != "scored" or not is_strict_finite_number(raw):
            return None
        value = float(raw)
        if internal in {"av_sync", "lip_sync"} and value < 0:
            return None
        return normalize_raw_mean(internal, value)

    @staticmethod
    def _lip_observations(row: Mapping[str, Any]) -> List[object]:
        if row.get("status") != "scored":
            return []
        details = row.get("raw_details")
        tracks = details.get("passing_tracks") if isinstance(details, Mapping) else None
        if not isinstance(tracks, list) or not tracks:
            return [None]
        observations: List[object] = []
        for track in tracks:
            if not isinstance(track, Mapping):
                observations.append(None)
                continue
            confidence = track.get("confidence")
            offset = track.get("offset_frames")
            if (
                not is_strict_finite_number(confidence)
                or float(confidence) < SYNCNET_LAUNCHER_CONFIDENCE_THRESHOLD
                or not is_strict_finite_number(offset)
            ):
                observations.append(None)
                continue
            observations.append(abs(float(offset)))
        return observations

    def _apply_dimension_result(
        self,
        internal: str,
        result: Mapping[str, Any],
        samples_by_id: Mapping[str, Sample],
        observations: Dict[str, List[object]],
    ) -> None:
        rows = result.get("records")
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            record_id = row.get("id")
            sample = samples_by_id.get(record_id) if isinstance(record_id, str) else None
            if sample is None:
                continue
            component = copy.deepcopy(dict(row))
            normalized = self._normalized_row_score(internal, row)
            component["normalized_score"] = normalized
            sample.metadata["avgen_bench"]["components"][internal] = component
            if normalized is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                field = SAMPLE_SCORE_FIELDS[internal]
                if field in QualityMetrics.model_fields:
                    setattr(sample.quality_metrics, field, normalized)
                else:
                    logger.warning("AVGen-Bench schema field is not declared: %s", field)
            if internal == "lip_sync":
                observations[internal].extend(self._lip_observations(row))
            elif row.get("status") == "scored":
                observations[internal].append(row.get("raw_score"))

    @staticmethod
    def _canonical_comparability(
        records: Sequence[Mapping[str, Any]],
        results: Mapping[str, Mapping[str, Any]],
        aggregate: Mapping[str, Any],
    ) -> Tuple[bool, bool]:
        expected = {(item.category, item.content) for item in canonical_prompts()}
        actual = {(str(item["category"]), str(item["content"])) for item in records}
        identity_complete = len(records) == len(expected) == 235 and actual == expected
        if not identity_complete:
            return False, False
        record_by_id = {str(row["id"]): row for row in records}
        missing_faces = set(missing_face_annotation_identities())
        complete = True
        for internal in DIMENSION_ORDER:
            result_rows = results.get(internal, {}).get("records")
            if not isinstance(result_rows, list) or len(result_rows) != len(records):
                complete = False
                continue
            returned = {str(row.get("id")): row for row in result_rows if isinstance(row, Mapping)}
            if set(returned) != set(record_by_id):
                complete = False
                continue
            for record_id, record in record_by_id.items():
                row = returned[record_id]
                status = row.get("status")
                identity = (record["category"], record["content"])
                if status in _BAD_COMPONENT_STATUSES:
                    complete = False
                elif internal == "face":
                    annotation = record.get("expected_faces")
                    if identity in missing_faces:
                        complete &= status == "missing_face_label"
                    elif isinstance(annotation, Mapping) and annotation.get("primary_faces", {}).get(
                        "max_ids"
                    ) == 0:
                        complete &= status == "ineligible"
                    else:
                        complete &= status == "scored"
                elif internal == "music" and record["category"] != "musical_instrument_tutorial":
                    complete &= status == "ineligible"
                elif internal in {"music", "text"}:
                    complete &= status in {"scored", "ineligible"}
                elif internal == "lip_sync":
                    complete &= status in {"scored", "ineligible"}
                else:
                    complete &= status == "scored"
        dimensions = aggregate.get("dimensions", {})
        all_dimensions_observed = all(
            isinstance(dimensions.get(key), Mapping) and dimensions[key].get("available") is True
            for key in DIMENSION_ORDER
        )
        return identity_complete, bool(complete and all_dimensions_observed)

    def _finalize_sample_statuses(self, samples: Sequence[Sample]) -> None:
        for sample in samples:
            metadata = sample.metadata.get("avgen_bench")
            if not isinstance(metadata, dict) or metadata.get("status") == "invalid_input":
                continue
            components = metadata.get("components", {})
            statuses = [
                value.get("status")
                for value in components.values()
                if isinstance(value, Mapping)
            ]
            metadata["status"] = (
                "evaluated"
                if len(statuses) == len(DIMENSION_ORDER)
                and not any(status in _BAD_COMPONENT_STATUSES for status in statuses)
                else "partial"
            )

    def post_process(self, all_samples: List[Sample]) -> None:
        self._reset_sample_outputs(all_samples)
        self.last_metadata = {}
        if self.test_mode:
            return

        records, samples_by_id, invalid_inputs = self._records_for_samples(all_samples)
        observations: Dict[str, List[object]] = {key: [] for key in DIMENSION_ORDER}
        source_summaries: Dict[str, Any] = {}
        component_results: Dict[str, Dict[str, Any]] = {}
        api_key = self._api_key()
        secret_values = [api_key] if api_key else []

        execution_records: Dict[str, List[Mapping[str, Any]]] = {}
        preflight_rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for internal in DIMENSION_ORDER:
            selected, skipped = self._dimension_plan(internal, records)
            execution_records[internal] = selected
            preflight_rows[internal] = skipped
            if records and not selected:
                source_key = SOURCE_DIMENSION_KEYS[internal]
                result = self._preflight_result(source_key, records, skipped)
                component_results[internal] = result
                source_summaries[source_key] = result["raw_summary"]
                self._apply_dimension_result(internal, result, samples_by_id, observations)

        # All five judge-backed dimensions are declared absent before any heavy
        # specialist runtime/model provisioning when credentials are unavailable.
        if api_key is None:
            for internal in DIMENSION_ORDER:
                if internal not in _API_DIMENSIONS:
                    continue
                if internal in component_results:
                    continue
                source_key = SOURCE_DIMENSION_KEYS[internal]
                result = self._missing_credentials_result(
                    source_key, execution_records[internal]
                )
                result = self._merge_preflight_rows(
                    source_key, records, result, preflight_rows[internal]
                )
                component_results[internal] = result
                source_summaries[source_key] = result["raw_summary"]
                self._apply_dimension_result(internal, result, samples_by_id, observations)

        models_dir = Path(self.config.get("models_dir", "models")).expanduser().resolve()
        run_root = (
            models_dir
            / ".ayase"
            / "work"
            / "avgen_bench"
            / ("run-" + uuid.uuid4().hex)
        )
        source_dir: Optional[Path] = None
        ledger: Optional[Dict[str, Any]] = None
        runtime = None
        assets_runtime = None
        if records:
            try:
                runtime = _load_runtime()
                source_dir = runtime.ensure_source(
                    models_dir,
                    source_override=self.config.get("source_root"),
                    test_mode=False,
                )
                run_root.mkdir(parents=True, exist_ok=False)
                ledger = runtime.stage_records(records, run_root / "inputs")
                assets_runtime = _load_assets_runtime()
            except Exception as exc:
                error = _bounded_error(exc, secret_values)
                logger.warning("AVGen-Bench preparation failed: %s", error)
                for internal in DIMENSION_ORDER:
                    if internal in component_results:
                        continue
                    source_key = SOURCE_DIMENSION_KEYS[internal]
                    result = self._failed_result(
                        source_key, execution_records[internal], error
                    )
                    result = self._merge_preflight_rows(
                        source_key, records, result, preflight_rows[internal]
                    )
                    component_results[internal] = result
                    source_summaries[source_key] = result["raw_summary"]
                    self._apply_dimension_result(internal, result, samples_by_id, observations)

        if ledger is not None and source_dir is not None and runtime is not None:
            for internal in DIMENSION_ORDER:
                if internal in component_results:
                    continue
                source_key = SOURCE_DIMENSION_KEYS[internal]
                worker_dir = run_root / "workers" / source_key.replace("-", "_")
                output_dir = run_root / "outputs" / source_key.replace("-", "_")
                selected_ids = {row["id"] for row in execution_records[internal]}
                selected_ledger_records = [
                    row for row in ledger["records"] if row["id"] in selected_ids
                ]
                try:
                    interpreter = runtime.ensure_runtime(source_key, models_dir, test_mode=False)
                    provisioned = assets_runtime.ensure_assets(
                        source_key,
                        models_dir,
                        overrides=self._asset_overrides(internal, assets_runtime),
                        test_mode=False,
                    )
                    if not isinstance(provisioned, Mapping):
                        raise TypeError("ensure_assets returned no asset mapping")
                    assets = _json_paths(provisioned.get("paths", {}))
                    identities = _json_paths(provisioned.get("asset_identities", {}))
                    config = self._worker_config(internal)
                    if internal in _FFMPEG_DIMENSIONS:
                        ffmpeg = runtime.runtime_ffmpeg(interpreter, models_dir)
                        config["ffmpeg_path"] = str(Path(ffmpeg).resolve())
                        if internal == "music":
                            assets.setdefault("ffmpeg_path", config["ffmpeg_path"])
                    request = {
                        "schema": 1,
                        "source_revision": SOURCE_REVISION,
                        "dimension": source_key,
                        "source_dir": str(source_dir.resolve()),
                        "models_dir": str(models_dir),
                        "work_dir": str(worker_dir.resolve()),
                        "videos_dir": ledger["videos_dir"],
                        "prompts_dir": ledger["prompts_dir"],
                        "face_prompts_dir": ledger["face_prompts_dir"],
                        "output_dir": str(output_dir.resolve()),
                        "run_tag": run_root.name,
                        "assets": assets,
                        "asset_identities": identities,
                        "config": config,
                        "records": selected_ledger_records,
                    }
                    result = runtime.execute_worker(
                        request,
                        interpreter,
                        api_key=api_key if internal in _API_DIMENSIONS else None,
                        timeout_s=float(self.config.get("worker_timeout_s", 14400)),
                        test_mode=False,
                    )
                except Exception as exc:
                    error = _bounded_error(exc, secret_values)
                    logger.warning("AVGen-Bench %s failed: %s", source_key, error)
                    result = self._failed_result(
                        source_key, execution_records[internal], error
                    )
                result = self._merge_preflight_rows(
                    source_key, records, result, preflight_rows[internal]
                )
                component_results[internal] = dict(result)
                source_summaries[source_key] = copy.deepcopy(result.get("raw_summary", {}))
                self._apply_dimension_result(internal, result, samples_by_id, observations)

        aggregate = aggregate_observations(
            observations, source_raw_summaries=source_summaries
        )
        mode = self.config.get("input_mode")
        identity_complete = False
        required_complete = False
        if mode == "canonical":
            identity_complete, required_complete = self._canonical_comparability(
                records, component_results, aggregate
            )
        canonical_summary = canonical_data_summary()
        metadata = {
            "schema": 1,
            "protocol_identity": PROTOCOL_IDENTITY,
            "source_protocol_identity": SOURCE_PROTOCOL_IDENTITY,
            "source_revision": SOURCE_REVISION,
            "launcher_profile": LAUNCHER_PROFILE,
            "declared_adaptations": list(PROTOCOL_ADAPTATIONS),
            "input_mode": mode,
            "input_sample_count": len(all_samples),
            "valid_input_count": len(records),
            "invalid_input_count": invalid_inputs,
            "run_dir": str(run_root) if run_root.exists() else None,
            "source_dir": str(source_dir) if source_dir is not None else None,
            "gemini_credentials_available": api_key is not None,
            "input_identities": [
                {
                    "id": row["id"],
                    "mode": row["mode"],
                    "category": row["category"],
                    "content": row["content"],
                    "prompt_sha256": hashlib.sha256(
                        row["prompt_text"].encode("utf-8")
                    ).hexdigest(),
                    "has_face_annotation": row.get("expected_faces") is not None,
                }
                for row in records
            ],
            "canonical_universe": canonical_summary,
            "canonical_identity_coverage": len(
                {
                    (row["category"], row["content"])
                    for row in records
                    if row.get("mode") == "canonical"
                }
            )
            / canonical_summary["total_prompts"],
            "canonical_complete_identity_coverage": identity_complete,
            "canonical_required_observations_complete": required_complete,
            "canonical_comparable": bool(identity_complete and required_complete),
            "aggregation": aggregate,
            "components": component_results,
        }
        self._finalize_sample_statuses(all_samples)
        self._publish_dataset(aggregate, metadata)

    def on_execute(self) -> None:
        self.last_metadata = {}

    def on_dispose(self) -> None:
        self.last_metadata = {}
        self._ml_available = False
        self._backend = None
        super().on_dispose()


__all__ = ["AVGenBenchModule"]

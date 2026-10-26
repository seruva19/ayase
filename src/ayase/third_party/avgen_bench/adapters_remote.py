"""Source-backed AVGen-Bench adapters for the five remote-judge dimensions.

This file is executed inside private specialist environments.  Keep its module-level
imports in the Python standard library: the runtime may intentionally contain none of
Ayase's host dependencies.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


SCHEMA_VERSION = 1
SOURCE_REVISION = "1049eabac472d479fe5feeb1ee202961f8e0982a"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
SUPPORTED_DIMENSIONS = ("Text", "Music", "Speech", "Hi-Phy", "Holistic")
DIMENSIONS = SUPPORTED_DIMENSIONS

_SOURCE_FILES = {
    "Text": "eval/Ocr/batch_eval.py",
    "Music": "eval/music_check/music_check_gemini.py",
    "Speech": "eval/speech/batch_eval.py",
    "Hi-Phy": "eval/gemini_phy/batch_eval.py",
    "Holistic": "eval/plot_matching/plot_matching_gemini.py",
}
_PROTOCOL_IDENTITIES = {
    "Text": "avgen-bench/text-ocr/original@1049eab",
    "Music": "avgen-bench/music-basic-pitch-raw-midi/original@1049eab",
    "Speech": "avgen-bench/speech-faster-whisper/original@1049eab",
    "Hi-Phy": "avgen-bench/hi-phy-dual-stage/original@1049eab",
    "Holistic": "avgen-bench/plot-matching/original@1049eab",
}
_BACKEND_IDENTITIES = {
    "Text": "PaddleOCR==3.3.2+Gemini-REST",
    "Music": "basic-pitch==0.4.0+Gemini-REST",
    "Speech": "faster-whisper==1.2.1+ctranslate2==4.6.3+Gemini-REST",
    "Hi-Phy": "Gemini-REST",
    "Holistic": "Gemini-REST",
}
_ROUTE_TIMEOUTS = {"Text": 300, "Music": 180, "Speech": 600, "Hi-Phy": 900, "Holistic": 900}
_VALID_STATUSES = {
    "scored",
    "ineligible",
    "missing_prompt",
    "missing_video",
    "missing_face_label",
    "no_face_detection",
    "failed",
}
_IMPORT_LOCK = threading.RLock()
_MISSING_MODULE = object()
_RAW_VALIDATION_POLICY = "avgen-strict-raw-json-v1"


class AdapterContractError(ValueError):
    """Raised when a runner request violates the public adapter contract."""


AdapterError = AdapterContractError


class RawJudgeValidationError(ValueError):
    """A Gemini response failed its source-prompt schema before source coercion."""

    def __init__(self, stage: str, raw_text: Any, reason: str, parsed: Any = None):
        self.stage = stage
        self.raw_text = raw_text if isinstance(raw_text, str) else str(raw_text)
        self.parsed = parsed
        self.reason = reason
        super().__init__(f"invalid raw Gemini response for {stage}: {reason}")


def _raw_object(raw_text: Any, stage: str) -> Dict[str, Any]:
    if not isinstance(raw_text, str):
        raise RawJudgeValidationError(stage, raw_text, "response text is not a string")
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    if not (text.startswith("{") and text.endswith("}")):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise RawJudgeValidationError(stage, raw_text, "response contains no JSON object")
        text = match.group(0)
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise RawJudgeValidationError(stage, raw_text, f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RawJudgeValidationError(stage, raw_text, "top-level JSON must be an object", parsed)
    _reject_nonfinite(parsed, stage, "$")
    return parsed


def _reject_nonfinite(value: Any, stage: str, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RawJudgeValidationError(stage, "", f"{path} must be finite", value)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, stage, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, stage, f"{path}[{index}]")


def _required(data: Mapping[str, Any], key: str, stage: str) -> Any:
    if key not in data:
        raise RawJudgeValidationError(stage, "", f"missing required field {key}", data)
    return data[key]


def _require_bool_field(data: Mapping[str, Any], key: str, stage: str) -> bool:
    value = _required(data, key, stage)
    if not isinstance(value, bool):
        raise RawJudgeValidationError(stage, "", f"{key} must be boolean", data)
    return value


def _require_int_score(data: Mapping[str, Any], key: str, stage: str) -> int:
    value = _required(data, key, stage)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RawJudgeValidationError(stage, "", f"{key} must be an integer", data)
    if not 0 <= value <= 100:
        raise RawJudgeValidationError(stage, "", f"{key} must be in [0, 100]", data)
    return value


def _require_confidence(data: Mapping[str, Any], key: str, stage: str) -> float:
    value = _required(data, key, stage)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RawJudgeValidationError(stage, "", f"{key} must be numeric", data)
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RawJudgeValidationError(stage, "", f"{key} must be finite and in [0, 1]", data)
    return number


def _require_container(
    data: Mapping[str, Any], key: str, expected: type, stage: str
) -> Any:
    value = _required(data, key, stage)
    if not isinstance(value, expected):
        raise RawJudgeValidationError(
            stage, "", f"{key} must be {expected.__name__}", data
        )
    return value


def _require_string(data: Mapping[str, Any], key: str, stage: str) -> str:
    value = _required(data, key, stage)
    if not isinstance(value, str):
        raise RawJudgeValidationError(stage, "", f"{key} must be a string", data)
    return value


def _validate_text_raw(data: Mapping[str, Any], stage: str) -> None:
    _require_bool_field(data, "prompt_requires_visible_text", stage)
    presence = _require_string(data, "text_presence", stage)
    if presence not in {"none", "noise", "incidental", "required"}:
        raise RawJudgeValidationError(stage, "", "text_presence has an invalid value", data)
    _require_bool_field(data, "missing_required_text", stage)
    contextual = _required(data, "incidental_text_is_contextual", stage)
    if contextual is not None and not isinstance(contextual, bool):
        raise RawJudgeValidationError(
            stage, "", "incidental_text_is_contextual must be boolean or null", data
        )
    _require_int_score(data, "overall_text_quality_score", stage)
    subscores = _require_container(data, "subscores", dict, stage)
    for key in ("legibility_accuracy", "temporal_stability", "spatial_stability", "completeness"):
        _require_int_score(subscores, key, stage)
    prompt_match = _required(subscores, "prompt_text_match", stage)
    if prompt_match is not None:
        _require_int_score(subscores, "prompt_text_match", stage)
    _require_confidence(data, "confidence", stage)
    _require_container(data, "top_issues", list, stage)
    _require_container(data, "good_tracks", list, stage)
    _require_string(data, "summary", stage)


def _validate_music_constraints_raw(data: Mapping[str, Any], stage: str) -> None:
    should_evaluate = _require_bool_field(data, "should_midi_eval", stage)
    _require_container(data, "constraints", dict, stage)
    if not should_evaluate:
        reason = _required(data, "skip_reason", stage)
        if not isinstance(reason, str) or not reason.strip():
            raise RawJudgeValidationError(
                stage, "", "skip_reason must be a non-empty string when ineligible", data
            )


def _validate_music_judge_raw(data: Mapping[str, Any], stage: str) -> None:
    _require_int_score(data, "overall_score", stage)
    _require_confidence(data, "confidence", stage)
    subscores = _require_container(data, "subscores", dict, stage)
    for key in ("chord", "sequence_or_scale"):
        value = _required(subscores, key, stage)
        if value is not None:
            _require_int_score(subscores, key, stage)
    _require_container(data, "interpretation", dict, stage)
    _require_container(data, "checks", list, stage)
    _require_container(data, "violations", list, stage)
    _require_string(data, "summary", stage)


def _validate_speech_raw(data: Mapping[str, Any], stage: str) -> None:
    score = _require_int_score(data, "score", stage)
    _require_bool_field(data, "explicit_speech_required", stage)
    match_type = _require_string(data, "match_type", stage)
    if match_type not in {"verbatim", "contextual"}:
        raise RawJudgeValidationError(stage, "", "match_type has an invalid value", data)
    passed = _require_bool_field(data, "pass", stage)
    if passed != (score >= 80):
        raise RawJudgeValidationError(stage, "", "pass is inconsistent with score", data)
    for key in (
        "required_speech_lines",
        "verbatim_match_details",
        "missing_lines",
        "extra_or_mismatched_segments",
    ):
        _require_container(data, key, list, stage)
    for key in ("normalized_transcript", "score_rationale", "suggested_fix"):
        _require_string(data, key, stage)


def _validate_hi_expectations_raw(data: Mapping[str, Any], stage: str) -> None:
    expectations = _require_container(data, "expectations", list, stage)
    for index, expectation in enumerate(expectations):
        if not isinstance(expectation, dict):
            raise RawJudgeValidationError(
                stage, "", f"expectations[{index}] must be an object", data
            )
        for key in ("id", "expectation"):
            if not isinstance(expectation.get(key), str) or not expectation[key].strip():
                raise RawJudgeValidationError(
                    stage, "", f"expectations[{index}].{key} must be a non-empty string", data
                )


def _validate_hi_semantic_raw(data: Mapping[str, Any], stage: str) -> None:
    _require_int_score(data, "semantic_physics_score", stage)
    _require_confidence(data, "confidence", stage)
    for key in ("event_log", "checks", "major_issues"):
        _require_container(data, key, list, stage)


def _validate_hi_visual_raw(data: Mapping[str, Any], stage: str) -> None:
    _require_int_score(data, "visual_physics_score", stage)
    _require_confidence(data, "confidence", stage)
    _require_bool_field(data, "cut_detected", stage)
    audit = _require_container(data, "audit_by_category", dict, stage)
    for key in (
        "kinematics",
        "gravity",
        "dynamics",
        "contact",
        "occlusion",
        "identity",
        "fluids_fire",
        "imaging",
    ):
        _require_container(audit, key, list, stage)
    _require_container(data, "visual_violations", list, stage)
    _require_container(data, "major_issues", list, stage)


def _validate_holistic_raw(data: Mapping[str, Any], stage: str) -> None:
    _require_int_score(data, "plot_alignment_score", stage)
    _require_confidence(data, "confidence", stage)
    subscores = _require_container(data, "subscores", dict, stage)
    for key in (
        "narrative_alignment",
        "shot_alignment",
        "visual_attribute_alignment",
        "audio_alignment",
    ):
        _require_int_score(subscores, key, stage)
    _require_container(data, "prompt_constraints", dict, stage)
    _require_container(data, "observed_event_log", list, stage)
    _require_container(data, "checks", list, stage)
    _require_container(data, "violations", list, stage)
    _require_string(data, "summary", stage)


_RAW_VALIDATORS = {
    "text.judge": _validate_text_raw,
    "music.constraints": _validate_music_constraints_raw,
    "music.judge": _validate_music_judge_raw,
    "speech.judge": _validate_speech_raw,
    "hi_phy.expectations": _validate_hi_expectations_raw,
    "hi_phy.semantic": _validate_hi_semantic_raw,
    "hi_phy.visual": _validate_hi_visual_raw,
    "holistic.judge": _validate_holistic_raw,
}


def _adapter_digest() -> str:
    with open(__file__, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    """Return JSON-safe evidence without allowing NaN/Infinity into the protocol."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return str(value)


def _safe_error(exc: BaseException) -> str:
    text = repr(exc)
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "<redacted>")
    text = re.sub(r"([?&]key=)[^&\s'\"]+", r"\1<redacted>", text, flags=re.IGNORECASE)
    return text[:2000]


def _mean(values: Iterable[float]) -> Optional[float]:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _bounded_workers(config: Mapping[str, Any], count: int) -> int:
    raw = config.get("workers", 4)
    if isinstance(raw, bool):
        raw = 4
    try:
        workers = int(raw)
    except (TypeError, ValueError):
        workers = 4
    return max(1, min(32, count or 1, workers))


def _timeout(config: Mapping[str, Any], dimension: str) -> int:
    raw = config.get("timeout_s", _ROUTE_TIMEOUTS[dimension])
    if isinstance(raw, bool):
        raise AdapterContractError("config.timeout_s must be a positive integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise AdapterContractError("config.timeout_s must be a positive integer") from exc
    if value <= 0:
        raise AdapterContractError("config.timeout_s must be a positive integer")
    return value


def _gemini_model(config: Mapping[str, Any]) -> str:
    value = config.get("gemini_model", DEFAULT_GEMINI_MODEL)
    if not isinstance(value, str) or not value.strip():
        raise AdapterContractError("config.gemini_model must be a non-empty string")
    return value


def _required_path(mapping: Mapping[str, Any], key: str, *, directory: bool) -> Path:
    raw = mapping.get(key)
    if not isinstance(raw, str) or not raw:
        raise AdapterContractError(f"assets.{key} is required")
    path = Path(raw).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise AdapterContractError(f"assets.{key} is not an existing {kind}: {path}")
    return path


def _source_path(source_dir: Path, relative: str) -> Path:
    root = source_dir.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AdapterContractError(f"source path escapes source_dir: {relative}") from exc
    if not path.is_file():
        raise AdapterContractError(f"missing pinned source file: {relative}")
    return path


@contextlib.contextmanager
def _temporary_environment(updates: Mapping[str, Optional[str]]) -> Iterable[None]:
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterable[None]:
    old = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _exec_module_from_path(module_name: str, source_file: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source module: {source_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_source_module(
    source_dir: Path, relative: str, namespace: str, *, bind_gemini: bool = False
) -> ModuleType:
    """Load one pinned source file under an isolated, content-addressed name."""

    source_file = _source_path(source_dir, relative)
    unique = hashlib.sha256(str(source_file).encode("utf-8")).hexdigest()[:12]
    module_name = f"_ayase_avgen_{namespace.lower().replace('-', '_')}_{unique}"
    root_text = str(source_dir)
    with _IMPORT_LOCK:
        inserted = root_text not in sys.path
        if inserted:
            sys.path.insert(0, root_text)
        old_no_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        previous_generic: Any = _MISSING_MODULE
        client_name: Optional[str] = None
        client_digest: Optional[str] = None
        try:
            if bind_gemini:
                client_path = _source_path(source_dir, "gemini_client.py")
                client_digest = hashlib.sha256(client_path.read_bytes()).hexdigest()
                client_name = f"_ayase_avgen_gemini_client_{unique}"
                pinned_client = _exec_module_from_path(client_name, client_path)
                previous_generic = sys.modules.get("gemini_client", _MISSING_MODULE)
                sys.modules["gemini_client"] = pinned_client
            module = _exec_module_from_path(module_name, source_file)
            if bind_gemini:
                module._ayase_pinned_gemini_client_sha256 = client_digest
            return module
        finally:
            if bind_gemini:
                if previous_generic is _MISSING_MODULE:
                    sys.modules.pop("gemini_client", None)
                else:
                    sys.modules["gemini_client"] = previous_generic
                if client_name is not None:
                    sys.modules.pop(client_name, None)
            sys.dont_write_bytecode = old_no_bytecode
            if inserted:
                try:
                    sys.path.remove(root_text)
                except ValueError:
                    pass


def _stage_for_instruction(module: ModuleType, dimension: str, instruction: Any) -> str:
    if dimension == "Text":
        return "text.judge"
    if dimension == "Speech":
        return "speech.judge"
    if dimension == "Holistic":
        return "holistic.judge"
    if dimension == "Music":
        prompts = module.PROMPT_VARIANTS["original"]
        stages = {
            prompts["stage1_system"]: "music.constraints",
            prompts["stage2_system"]: "music.judge",
        }
    elif dimension == "Hi-Phy":
        prompts = module.PROMPT_VARIANTS["original"]
        stages = {
            prompts["expect_system"]: "hi_phy.expectations",
            prompts["semantic_system"]: "hi_phy.semantic",
            prompts["visual_system"]: "hi_phy.visual",
        }
    else:
        raise AdapterContractError(f"no raw response schema for {dimension}")
    stage = stages.get(instruction)
    if stage is None:
        raise AdapterContractError(f"unrecognized {dimension} source judge prompt")
    return stage


def _install_raw_validation(module: ModuleType, dimension: str) -> None:
    original = module.generate_content_text
    state = threading.local()

    def begin() -> None:
        state.events = []

    def consume() -> List[Dict[str, Any]]:
        events = list(getattr(state, "events", []))
        state.events = []
        return events

    def validated_generate_content_text(*args: Any, **kwargs: Any) -> str:
        raw_text = original(*args, **kwargs)
        stage = _stage_for_instruction(module, dimension, kwargs.get("system_instruction"))
        event: Dict[str, Any] = {
            "stage": stage,
            "policy": _RAW_VALIDATION_POLICY,
            "raw_text": raw_text if isinstance(raw_text, str) else str(raw_text),
        }
        try:
            parsed = _raw_object(raw_text, stage)
            _RAW_VALIDATORS[stage](parsed, stage)
            event.update(valid=True, parsed=_json_safe(parsed))
            state.events = [*getattr(state, "events", []), event]
            return raw_text
        except RawJudgeValidationError as exc:
            if not exc.raw_text:
                exc.raw_text = event["raw_text"]
            if exc.parsed is None:
                try:
                    exc.parsed = parsed
                except UnboundLocalError:
                    pass
            event.update(
                valid=False,
                parsed=_json_safe(exc.parsed),
                validation_error=str(exc),
            )
            state.events = [*getattr(state, "events", []), event]
            raise

    module.generate_content_text = validated_generate_content_text
    module._ayase_raw_validation_begin = begin
    module._ayase_raw_validation_consume = consume


def _raw_begin(module: ModuleType) -> None:
    begin = getattr(module, "_ayase_raw_validation_begin", None)
    if callable(begin):
        begin()


def _raw_consume(module: ModuleType) -> List[Dict[str, Any]]:
    consume = getattr(module, "_ayase_raw_validation_consume", None)
    return consume() if callable(consume) else []


def _attach_raw_evidence(
    details: Mapping[str, Any], module: ModuleType, events: List[Dict[str, Any]]
) -> Dict[str, Any]:
    evidence = dict(details)
    evidence["_adapter_raw_validation"] = {
        "policy": _RAW_VALIDATION_POLICY,
        "pinned_gemini_client_sha256": getattr(
            module, "_ayase_pinned_gemini_client_sha256", None
        ),
        "events": events,
    }
    return evidence


def _load_text_module(source_dir: Path, det_dir: Path, rec_dir: Path) -> ModuleType:
    """Inject the pinned Paddle model directories before OCR's import-time constructor."""

    paddleocr = importlib.import_module("paddleocr")
    original = paddleocr.PaddleOCR

    def configured_paddle_ocr(*args: Any, **kwargs: Any) -> Any:
        kwargs["text_detection_model_dir"] = str(det_dir)
        kwargs["text_recognition_model_dir"] = str(rec_dir)
        return original(*args, **kwargs)

    paddleocr.PaddleOCR = configured_paddle_ocr
    try:
        return _load_source_module(
            source_dir, _SOURCE_FILES["Text"], "text", bind_gemini=True
        )
    finally:
        paddleocr.PaddleOCR = original


def _record_token(record: Mapping[str, Any]) -> str:
    identity = f"{record.get('category', '')}\0{record.get('safe_content', '')}\0{record['id']}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _write_evidence(output_dir: Path, dimension: str, record: Mapping[str, Any], details: Any) -> str:
    target_dir = output_dir / dimension.lower().replace("-", "_")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{_record_token(record)}.json"
    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(json.dumps(_json_safe(details), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return str(target.resolve())


def _new_result(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "status": "failed",
        "raw_score": None,
        "raw_details": {},
        "eligible": None,
    }


def _preflight(record: Mapping[str, Any], result: MutableMapping[str, Any]) -> Optional[Path]:
    prompt = record.get("prompt_text")
    if not isinstance(prompt, str) or not prompt.strip():
        result["status"] = "missing_prompt"
        return None
    raw_path = record.get("staged_video_path") or record.get("video_path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_file():
        result["status"] = "missing_video"
        return None
    return Path(raw_path).resolve()


def _source_identity(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "category": record.get("category"),
        "safe_content": record.get("safe_content"),
        "mode": record.get("mode"),
    }


def _set_failed(
    result: MutableMapping[str, Any],
    exc: BaseException,
    *,
    details: Optional[Mapping[str, Any]] = None,
    record: Optional[Mapping[str, Any]] = None,
    output_dir: Optional[Path] = None,
    dimension: Optional[str] = None,
) -> None:
    result["status"] = "failed"
    result["raw_score"] = None
    result["eligible"] = None
    result["error"] = _safe_error(exc)
    if details is not None:
        evidence = dict(details)
        if record is not None:
            evidence["_adapter_source_identity"] = _source_identity(record)
        evidence["_adapter_validation_error"] = result["error"]
        result["raw_details"] = _json_safe(evidence)
        if record is not None and output_dir is not None and dimension is not None:
            result["evidence_path"] = _write_evidence(
                output_dir, dimension, record, evidence
            )


def _require_dict(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} returned {type(value).__name__}, expected dict")
    return value


def _require_score(details: Mapping[str, Any], key: str) -> float:
    if key not in details:
        raise ValueError(f"source result missing {key}")
    score = _finite_number(details[key])
    if score is None:
        raise ValueError(f"source result has invalid {key}")
    return score


def _finalize_scored(
    result: MutableMapping[str, Any],
    record: Mapping[str, Any],
    details: Mapping[str, Any],
    score: float,
    output_dir: Path,
    dimension: str,
) -> None:
    evidence = dict(details)
    evidence["_adapter_source_identity"] = _source_identity(record)
    result.update(
        status="scored",
        raw_score=score,
        raw_details=_json_safe(evidence),
        eligible=True,
        evidence_path=_write_evidence(output_dir, dimension, record, evidence),
    )


def _run_text(context: Dict[str, Any], pending: List[Tuple[int, Mapping[str, Any], Path]]) -> Dict[str, Any]:
    assets = context["assets"]
    det_dir = _required_path(assets, "ocr_det_dir", directory=True)
    rec_dir = _required_path(assets, "ocr_rec_dir", directory=True)
    config = context["config"]
    timeout_s = _timeout(config, "Text")
    model = _gemini_model(config)
    env = {
        "GEMINI_TIMEOUT_S": str(timeout_s),
        "OCR_PROMPT_VARIANT": "original",
    }
    configured_device = config.get("device", "auto")
    if configured_device not in (None, "auto"):
        if not isinstance(configured_device, str) or not configured_device:
            raise AdapterContractError("Text config.device must be 'auto' or a Paddle device")
        env["OCR_DEVICE"] = "gpu:0" if configured_device == "cuda" else configured_device
    with _temporary_environment(env):
        module = _load_text_module(context["source_dir"], det_dir, rec_dir)
    module.MODEL_NAME = model
    module.TIMEOUT_S = timeout_s
    module.PROMPT_VARIANT = "original"
    _install_raw_validation(module, "Text")

    payloads: Dict[int, Dict[str, Any]] = {}
    for index, record, video_path in pending:
        try:
            payload_path = context["work_dir"] / "text" / f"{_record_token(record)}.ocr.json"
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payloads[index] = _require_dict(
                module.run_ocr_only(str(video_path), str(payload_path)), "run_ocr_only"
            )
        except Exception as exc:
            _set_failed(context["results"][index], exc)

    def judge(
        item: Tuple[int, Mapping[str, Any], Path]
    ) -> Tuple[
        int,
        Optional[Dict[str, Any]],
        Optional[BaseException],
        List[Dict[str, Any]],
    ]:
        index, record, _ = item
        if index not in payloads:
            return index, None, None, []
        _raw_begin(module)
        try:
            verdict = _require_dict(
                module.run_gemini_only(record["prompt_text"], payloads[index]), "run_gemini_only"
            )
            return index, verdict, None, _raw_consume(module)
        except Exception as exc:
            return index, None, exc, _raw_consume(module)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_bounded_workers(config, len(payloads))
    ) as executor:
        judged = list(executor.map(judge, pending))

    eligible_scores: List[float] = []
    valid_count = 0
    presence_counts: Dict[str, int] = {}
    for index, verdict, error, raw_events in judged:
        if error is not None:
            record = context["records"][index]
            _set_failed(
                context["results"][index],
                error,
                details=_attach_raw_evidence(
                    {"error_stage": "gemini", "tracks_summary": payloads[index]},
                    module,
                    raw_events,
                ),
                record=record,
                output_dir=context["output_dir"],
                dimension="Text",
            )
            continue
        if verdict is None:
            continue
        result = context["results"][index]
        record = context["records"][index]
        try:
            score = _require_score(verdict, "overall_text_quality_score")
            valid_count += 1
            presence = str(verdict.get("text_presence", "none") or "none").strip().lower()
            presence_counts[presence] = presence_counts.get(presence, 0) + 1
            eligible = verdict.get("prompt_requires_visible_text") is True or presence == "incidental"
            details = _attach_raw_evidence(verdict, module, raw_events)
            details["tracks_summary"] = payloads[index]
            details["_adapter_source_identity"] = _source_identity(record)
            result["raw_details"] = _json_safe(details)
            result["evidence_path"] = _write_evidence(
                context["output_dir"], "Text", record, details
            )
            if eligible:
                result.update(status="scored", raw_score=score, eligible=True)
                eligible_scores.append(score)
            else:
                result.update(status="ineligible", raw_score=None, eligible=False)
        except Exception as exc:
            _set_failed(
                result,
                exc,
                details=verdict,
                record=record,
                output_dir=context["output_dir"],
                dimension="Text",
            )

    return {
        "count_scored": valid_count,
        "count_eligible_for_summary": len(eligible_scores),
        "mean_score": _mean(eligible_scores),
        "avg_score": _mean(eligible_scores),
        "text_presence_counts": presence_counts,
        "summary_inclusion_rule": (
            "include sample if prompt_requires_visible_text == true or text_presence == 'incidental'"
        ),
    }


def _run_music(context: Dict[str, Any], pending: List[Tuple[int, Mapping[str, Any], Path]]) -> Dict[str, Any]:
    selected: List[Tuple[int, Mapping[str, Any], Path]] = []
    source_filtered = 0
    for index, record, video_path in pending:
        if (
            record.get("mode") == "canonical"
            and record.get("category") != "musical_instrument_tutorial"
        ):
            source_filtered += 1
            context["results"][index].update(
                status="ineligible",
                raw_score=None,
                eligible=False,
                raw_details={
                    "source_selection": "canonical launcher selects musical_instrument_tutorial only",
                    "_adapter_source_identity": _source_identity(record),
                },
            )
        else:
            selected.append((index, record, video_path))
    if not selected:
        return {
            "num_total": 0,
            "num_ok": 0,
            "num_prompt_skipped": 0,
            "num_midi_evaluable": 0,
            "total_score": 0.0,
            "mean_score": 0.0,
            "confidence_weighted_mean_score": 0.0,
            "adapter_source_filtered": source_filtered,
        }

    model_path = _required_path(context["assets"], "basic_pitch_model", directory=False)
    config = context["config"]
    timeout_s = _timeout(config, "Music")
    model = _gemini_model(config)
    temp_dir = (context["work_dir"] / "music" / "temp").resolve()
    cache_dir = (context["work_dir"] / "music" / "constraints").resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = context["assets"].get("ffmpeg_path") or config.get("ffmpeg_path")
    env = {
        "MUSIC_TEMP_DIR": str(temp_dir),
        "MUSIC_PROMPT_VARIANT": "original",
        "GEMINI_TIMEOUT_S": str(timeout_s),
        "IMAGEIO_FFMPEG_EXE": str(Path(ffmpeg).resolve()) if ffmpeg else None,
        "FFMPEG_BINARY": str(Path(ffmpeg).resolve()) if ffmpeg else None,
    }
    with _temporary_environment(env):
        module = _load_source_module(
            context["source_dir"], _SOURCE_FILES["Music"], "music", bind_gemini=True
        )
    module.MODEL_NAME = model
    module.DEFAULT_TIMEOUT_S = timeout_s
    module.PROMPT_VARIANT = "original"
    module.TEMP_DIR = temp_dir
    module.CHORD_ONSET_WINDOW_S = 0.08
    module.MAX_EVENTS_TO_SEND = 600
    _install_raw_validation(module, "Music")

    source_transcriber = module.BasicPitchTranscriber

    class PinnedBasicPitchTranscriber(source_transcriber):
        def __init__(self, temp_dir: Path = temp_dir):
            super().__init__(temp_dir=temp_dir)
            self.model_path = str(model_path)

    module.BasicPitchTranscriber = PinnedBasicPitchTranscriber

    scores: List[float] = []
    confidences: List[Tuple[float, float]] = []
    skipped = 0
    with _working_directory(context["work_dir"] / "music"):
        for index, record, video_path in selected:
            result = context["results"][index]
            details: Optional[Dict[str, Any]] = None
            raw_events: List[Dict[str, Any]] = []
            _raw_begin(module)
            try:
                evidence_target = context["output_dir"] / "music" / f"{_record_token(record)}.json"
                details = _require_dict(
                    module.judge_video_music_raw_midi(
                        video_path=str(video_path),
                        prompt_text=record["prompt_text"],
                        out_path=str(evidence_target),
                        constraints_cache_dir=str(cache_dir),
                        timeout_s=timeout_s,
                    ),
                    "judge_video_music_raw_midi",
                )
                raw_events = _raw_consume(module)
                details = _attach_raw_evidence(details, module, raw_events)
                raw_source_score = _require_score(details, "overall_score")
                evidence = dict(details)
                evidence["_adapter_source_identity"] = _source_identity(record)
                result["raw_details"] = _json_safe(evidence)
                result["evidence_path"] = _write_evidence(
                    context["output_dir"], "Music", record, evidence
                )
                if details.get("skipped") is True:
                    skipped += 1
                    result.update(status="ineligible", raw_score=None, eligible=False)
                    continue
                if details.get("skipped") is not False:
                    raise ValueError("source music result missing boolean skipped field")
                result.update(status="scored", raw_score=raw_source_score, eligible=True)
                scores.append(raw_source_score)
                confidence = _finite_number(details.get("confidence"))
                confidences.append((raw_source_score, confidence if confidence is not None else 0.0))
            except Exception as exc:
                if not raw_events:
                    raw_events = _raw_consume(module)
                failed_details = _attach_raw_evidence(
                    details or {}, module, raw_events
                )
                _set_failed(
                    result,
                    exc,
                    details=failed_details,
                    record=record,
                    output_dir=context["output_dir"],
                    dimension="Music",
                )

    total_weight = sum(max(0.0, min(1.0, confidence)) for _, confidence in confidences)
    weighted = (
        sum(score * max(0.0, min(1.0, confidence)) for score, confidence in confidences)
        / total_weight
        if total_weight > 0
        else 0.0
    )
    return {
        "num_total": len(selected),
        "num_ok": len(scores) + skipped,
        "num_prompt_skipped": skipped,
        "num_midi_evaluable": len(scores),
        "total_score": sum(scores),
        # Preserve released source behavior here; public eligibility remains strict above.
        "mean_score": _mean(scores) if scores else 0.0,
        "confidence_weighted_mean_score": weighted,
        "adapter_source_filtered": source_filtered,
    }


def _run_speech(context: Dict[str, Any], pending: List[Tuple[int, Mapping[str, Any], Path]]) -> Dict[str, Any]:
    model_dir = _required_path(context["assets"], "whisper_model_dir", directory=True)
    config = context["config"]
    timeout_s = _timeout(config, "Speech")
    gemini_model = _gemini_model(config)
    with _temporary_environment({"SPEECH_PROMPT_VARIANT": "original"}):
        module = _load_source_module(
            context["source_dir"], _SOURCE_FILES["Speech"], "speech", bind_gemini=True
        )
    module.PROMPT_VARIANT = "original"
    _install_raw_validation(module, "Speech")

    configured_device = config.get("device", "auto")
    if configured_device == "auto":
        device = "cuda" if module.torch.cuda.is_available() else "cpu"
    elif configured_device in ("cpu", "cuda"):
        device = configured_device
    else:
        raise AdapterContractError("Speech config.device must be 'auto', 'cpu', or 'cuda'")
    requested_compute = config.get("compute_type", "float16")
    if not isinstance(requested_compute, str) or not requested_compute:
        raise AdapterContractError("Speech config.compute_type must be a non-empty string")
    compute_type = requested_compute if device == "cuda" else "int8"
    whisper = module.WhisperModel(str(model_dir), device=device, compute_type=compute_type)

    transcripts: Dict[int, Dict[str, Any]] = {}
    for index, record, video_path in pending:
        try:
            transcribed = _require_dict(
                module.transcribe_with_faster_whisper(
                    whisper,
                    str(video_path),
                    language="en",
                    beam_size=1,
                    temperature=0.0,
                    vad_filter=True,
                    vad_min_silence_ms=300,
                    vad_speech_pad_ms=500,
                ),
                "transcribe_with_faster_whisper",
            )
            transcripts[index] = transcribed
        except Exception as exc:
            _set_failed(context["results"][index], exc)

    api_key = module.resolve_api_key(required=False) or ""
    if not api_key:
        missing_key = RuntimeError("Missing Gemini API key.")
        for index in transcripts:
            _set_failed(context["results"][index], missing_key)
        transcripts = {}

    retries = config.get("gemini_retries", 3)
    if isinstance(retries, bool):
        retries = 3
    retries = max(1, min(10, int(retries)))

    def judge(
        item: Tuple[int, Mapping[str, Any], Path]
    ) -> Tuple[
        int,
        Optional[Dict[str, Any]],
        Optional[BaseException],
        List[Dict[str, Any]],
    ]:
        index, record, _ = item
        if index not in transcripts:
            return index, None, None, []
        _raw_begin(module)
        try:
            verdict = _require_dict(
                module.evaluate_speech_with_gemini(
                    generation_prompt=record["prompt_text"],
                    transcript_text=str(transcripts[index].get("text", "")),
                    model_name=gemini_model,
                    api_key=api_key,
                    timeout_s=timeout_s,
                    max_retries=retries,
                ),
                "evaluate_speech_with_gemini",
            )
            return index, verdict, None, _raw_consume(module)
        except Exception as exc:
            return index, None, exc, _raw_consume(module)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_bounded_workers(config, len(transcripts))
    ) as executor:
        judged = list(executor.map(judge, pending))

    scores: List[float] = []
    by_category: Dict[str, List[float]] = {}
    video_paths = {index: video_path for index, _, video_path in pending}
    for index, verdict, error, raw_events in judged:
        if error is not None:
            record = context["records"][index]
            _set_failed(
                context["results"][index],
                error,
                details=_attach_raw_evidence(
                    {
                        "error_stage": "gemini",
                        "prompt": record["prompt_text"],
                        "transcription": transcripts[index],
                    },
                    module,
                    raw_events,
                ),
                record=record,
                output_dir=context["output_dir"],
                dimension="Speech",
            )
            continue
        if verdict is None:
            continue
        record = context["records"][index]
        result = context["results"][index]
        try:
            score = _require_score(verdict, "score")
            transcription = transcripts[index]
            details = {
                "video_path": str(video_paths[index]),
                "category": record.get("category"),
                "content": record.get("content"),
                "prompt": record["prompt_text"],
                "transcript": str(transcription.get("text", "")),
                "transcription": transcription,
                "whisper_meta": {
                    "engine": "faster-whisper",
                    "language": transcription.get("language"),
                    "language_probability": transcription.get("language_probability"),
                    "duration": transcription.get("duration"),
                    "beam_size": 1,
                    "temperature": 0.0,
                    "vad_filter": True,
                    "vad_min_silence_ms": 300,
                    "vad_speech_pad_ms": 500,
                    "compute_type": compute_type,
                    "device": device,
                },
                "judgement": verdict,
                "_adapter_source_identity": _source_identity(record),
            }
            details = _attach_raw_evidence(details, module, raw_events)
            _finalize_scored(result, record, details, score, context["output_dir"], "Speech")
            scores.append(score)
            category = str(record.get("category") or "unknown")
            by_category.setdefault(category, []).append(score)
        except Exception as exc:
            failed_details = None
            if verdict is not None and index in transcripts:
                failed_details = {
                    "prompt": record["prompt_text"],
                    "transcription": transcripts[index],
                    "judgement": verdict,
                }
            _set_failed(
                result,
                exc,
                details=failed_details,
                record=record,
                output_dir=context["output_dir"],
                dimension="Speech",
            )

    return {
        "whisper_model": str(model_dir),
        "whisper_language": "en",
        "gemini_model": gemini_model,
        "count_total_videos_found": len(pending),
        "count_transcribed": len(transcripts),
        "count_scored": len(scores),
        "avg_score": _mean(scores),
        "by_category": {
            category: {
                "n": len(category_scores),
                "avg_score": _mean(category_scores),
                "pass_rate": _mean([1.0 if score >= 80 else 0.0 for score in category_scores]),
            }
            for category, category_scores in by_category.items()
        },
    }


def _run_hi_phy(context: Dict[str, Any], pending: List[Tuple[int, Mapping[str, Any], Path]]) -> Dict[str, Any]:
    config = context["config"]
    timeout_s = _timeout(config, "Hi-Phy")
    model = _gemini_model(config)
    with _temporary_environment({"GEMINI_TIMEOUT_S": str(timeout_s)}):
        module = _load_source_module(
            context["source_dir"], _SOURCE_FILES["Hi-Phy"], "hi_phy", bind_gemini=True
        )
    module.PROMPT_VARIANT = "original"
    _install_raw_validation(module, "Hi-Phy")
    cache_path = context["work_dir"] / "hi_phy" / "expectations_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(
        item: Tuple[int, Mapping[str, Any], Path]
    ) -> Tuple[
        int,
        Optional[Dict[str, Any]],
        Optional[BaseException],
        List[Dict[str, Any]],
    ]:
        index, record, video_path = item
        _raw_begin(module)
        try:
            details = _require_dict(
                module.run_two_stage_with_dual_stage2(
                    video_path=str(video_path),
                    prompt_text=record["prompt_text"],
                    model_name=model,
                    expectations_cache_path=str(cache_path),
                    save_result_path=None,
                    timeout_s=timeout_s,
                    weights=(0.5, 0.5),
                ),
                "run_two_stage_with_dual_stage2",
            )
            return index, details, None, _raw_consume(module)
        except Exception as exc:
            return index, None, exc, _raw_consume(module)

    # The released source explicitly falls back to an unlocked shared JSON cache when
    # filelock is absent.  Keep that fallback sequential so parallel writes cannot lose
    # prompt expectations.
    hi_workers = (
        _bounded_workers(config, len(pending))
        if getattr(module, "FileLock", None) is not None
        else 1
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=hi_workers) as executor:
        evaluated = list(executor.map(evaluate, pending))

    scores: List[float] = []
    by_category: Dict[str, List[float]] = {}
    for index, details, error, raw_events in evaluated:
        result = context["results"][index]
        if error is not None:
            _set_failed(
                result,
                error,
                details=_attach_raw_evidence({}, module, raw_events),
                record=context["records"][index],
                output_dir=context["output_dir"],
                dimension="Hi-Phy",
            )
            continue
        try:
            assert details is not None
            details = _attach_raw_evidence(details, module, raw_events)
            score = _require_score(details, "overall_score")
            record = context["records"][index]
            _finalize_scored(result, record, details, score, context["output_dir"], "Hi-Phy")
            scores.append(score)
            by_category.setdefault(str(record.get("category") or "unknown"), []).append(score)
        except Exception as exc:
            _set_failed(
                result,
                exc,
                details=details,
                record=context["records"][index],
                output_dir=context["output_dir"],
                dimension="Hi-Phy",
            )
    return {
        "mean_overall": _mean(scores),
        "n": len(scores),
        "by_folder": {
            category: {"n": len(values), "mean_overall": _mean(values)}
            for category, values in by_category.items()
        },
    }


def _run_holistic(context: Dict[str, Any], pending: List[Tuple[int, Mapping[str, Any], Path]]) -> Dict[str, Any]:
    config = context["config"]
    timeout_s = _timeout(config, "Holistic")
    model = _gemini_model(config)
    env = {
        "PLOT_MATCHING_GEMINI_MODEL": model,
        "PLOT_MATCHING_PROMPT_VARIANT": "original",
        "GEMINI_TIMEOUT_S": str(timeout_s),
    }
    with _temporary_environment(env):
        module = _load_source_module(
            context["source_dir"],
            _SOURCE_FILES["Holistic"],
            "holistic",
            bind_gemini=True,
        )
    module.MODEL_NAME = model
    module.PROMPT_VARIANT = "original"
    _install_raw_validation(module, "Holistic")

    def evaluate(
        item: Tuple[int, Mapping[str, Any], Path]
    ) -> Tuple[
        int,
        Optional[Dict[str, Any]],
        Optional[BaseException],
        List[Dict[str, Any]],
    ]:
        index, record, video_path = item
        _raw_begin(module)
        try:
            details = _require_dict(
                module.judge_plot_matching(
                    str(video_path), record["prompt_text"], timeout_s=timeout_s
                ),
                "judge_plot_matching",
            )
            return index, details, None, _raw_consume(module)
        except Exception as exc:
            return index, None, exc, _raw_consume(module)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_bounded_workers(config, len(pending))
    ) as executor:
        evaluated = list(executor.map(evaluate, pending))

    scores: List[float] = []
    by_category: Dict[str, List[float]] = {}
    for index, details, error, raw_events in evaluated:
        result = context["results"][index]
        if error is not None:
            _set_failed(
                result,
                error,
                details=_attach_raw_evidence({}, module, raw_events),
                record=context["records"][index],
                output_dir=context["output_dir"],
                dimension="Holistic",
            )
            continue
        try:
            assert details is not None
            details = _attach_raw_evidence(details, module, raw_events)
            score = _require_score(details, "plot_alignment_score")
            subscores = details.get("subscores")
            required = (
                "narrative_alignment",
                "shot_alignment",
                "visual_attribute_alignment",
                "audio_alignment",
            )
            if not isinstance(subscores, dict) or any(
                _finite_number(subscores.get(name)) is None for name in required
            ):
                raise ValueError("source plot result missing one or more finite subscores")
            if _finite_number(details.get("confidence")) is None:
                raise ValueError("source plot result has invalid confidence")
            record = context["records"][index]
            _finalize_scored(result, record, details, score, context["output_dir"], "Holistic")
            scores.append(score)
            by_category.setdefault(str(record.get("category") or "unknown"), []).append(score)
        except Exception as exc:
            _set_failed(
                result,
                exc,
                details=details,
                record=context["records"][index],
                output_dir=context["output_dir"],
                dimension="Holistic",
            )
    return {
        "summaries_by_category": [
            {
                "category": category,
                "count_scored": len(values),
                "mean_plot_alignment_score": _mean(values),
            }
            for category, values in sorted(by_category.items())
        ],
        "overall": {
            "count_scored_total": len(scores),
            "mean_plot_alignment_score_total": _mean(scores),
        },
    }


_HANDLERS = {
    "Text": _run_text,
    "Music": _run_music,
    "Speech": _run_speech,
    "Hi-Phy": _run_hi_phy,
    "Holistic": _run_holistic,
}


def _validate_request(request: Mapping[str, Any]) -> Tuple[str, List[Mapping[str, Any]]]:
    if not isinstance(request, dict):
        raise AdapterContractError("request must be a dict")
    if request.get("schema") != SCHEMA_VERSION:
        raise AdapterContractError("request.schema must equal 1")
    dimension = request.get("dimension")
    if dimension not in SUPPORTED_DIMENSIONS:
        raise AdapterContractError(f"unsupported remote dimension: {dimension!r}")
    records = request.get("records")
    if not isinstance(records, list):
        raise AdapterContractError("request.records must be a list")
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise AdapterContractError(f"records[{index}] must be a dict")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise AdapterContractError(f"records[{index}].id must be a non-empty string")
        if record_id in seen:
            raise AdapterContractError(f"duplicate record id: {record_id}")
        seen.add(record_id)
    return dimension, records


def _build_counts(
    results: List[Mapping[str, Any]], records: List[Mapping[str, Any]]
) -> Dict[str, int]:
    counts = {
        "requested": len(results),
        "located": sum(
            Path(str(record.get("staged_video_path") or record.get("video_path") or "")).is_file()
            for record in records
        ),
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
    for result in results:
        status = result["status"]
        eligible = result.get("eligible")
        if eligible is True:
            counts["eligibility_known_true"] += 1
        elif eligible is False:
            counts["eligibility_known_false"] += 1
        else:
            counts["eligibility_unknown"] += 1
        if status == "scored":
            counts["scored"] += 1
        elif status == "failed":
            counts["failed"] += 1
        elif status in (
            "ineligible",
            "missing_prompt",
            "missing_video",
            "missing_face_label",
            "no_face_detection",
        ):
            counts[status] += 1
    counts["summary_eligible_numeric_count"] = counts["scored"]
    return counts


def _asset_identities(dimension: str, assets: Mapping[str, Any]) -> Dict[str, Any]:
    keys = {
        "Text": ("ocr_det_dir", "ocr_rec_dir"),
        "Music": ("basic_pitch_model", "ffmpeg_path"),
        "Speech": ("whisper_model_dir",),
        "Hi-Phy": (),
        "Holistic": (),
    }[dimension]
    result: Dict[str, Any] = {}
    for key in keys:
        value = assets.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value).expanduser().resolve()
        identity: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.is_file():
            identity["size"] = path.stat().st_size
        result[key] = identity
    return result


def run_dimension(request: dict) -> dict:
    """Run one complete remote AVGen dimension and return protocol schema 1."""

    dimension, records = _validate_request(request)
    source_dir_raw = request.get("source_dir")
    if not isinstance(source_dir_raw, str) or not Path(source_dir_raw).is_absolute():
        raise AdapterContractError("request.source_dir must be an absolute path")
    source_dir = Path(source_dir_raw).resolve()
    if not source_dir.is_dir():
        raise AdapterContractError(f"request.source_dir does not exist: {source_dir}")
    requested_revision = request.get("source_revision")
    if requested_revision is not None and requested_revision != SOURCE_REVISION:
        raise AdapterContractError("request source_revision does not match the pinned adapter source")

    def request_path(name: str) -> Path:
        raw = request.get(name)
        if not isinstance(raw, str) or not raw:
            raise AdapterContractError(f"request.{name} must be a non-empty path string")
        path = Path(raw).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    work_dir = request_path("work_dir")
    output_dir = request_path("output_dir")
    config = request.get("config", {})
    assets = request.get("assets", {})
    if not isinstance(config, dict) or not isinstance(assets, dict):
        raise AdapterContractError("request.config and request.assets must be dicts")

    results = [_new_result(record) for record in records]
    pending: List[Tuple[int, Mapping[str, Any], Path]] = []
    for index, record in enumerate(records):
        video_path = _preflight(record, results[index])
        if video_path is not None:
            pending.append((index, record, video_path))

    context = {
        "source_dir": source_dir,
        "work_dir": work_dir,
        "output_dir": output_dir,
        "config": config,
        "assets": assets,
        "records": records,
        "results": results,
    }
    raw_summary: Dict[str, Any] = {}
    if pending:
        try:
            raw_summary = _HANDLERS[dimension](context, pending)
        except Exception as exc:
            for index, _, _ in pending:
                if results[index]["status"] == "failed" and "error" not in results[index]:
                    _set_failed(results[index], exc)

    for result in results:
        if result["status"] not in _VALID_STATUSES:
            raise RuntimeError(f"internal invalid status: {result['status']}")
        score = result.get("raw_score")
        if score is not None and _finite_number(score) is None:
            raise RuntimeError("internal non-finite or boolean raw_score")

    provided_asset_identities = request.get("asset_identities")
    response = {
        "schema": SCHEMA_VERSION,
        "dimension": dimension,
        "source_revision": SOURCE_REVISION,
        "protocol_identity": _PROTOCOL_IDENTITIES[dimension],
        "backend_identity": _BACKEND_IDENTITIES[dimension],
        "adapter_digest": _adapter_digest(),
        "asset_identities": (
            _json_safe(provided_asset_identities)
            if isinstance(provided_asset_identities, Mapping)
            else _asset_identities(dimension, assets)
        ),
        "records": results,
        "raw_summary": _json_safe(raw_summary),
        "counts": _build_counts(results, records),
    }
    return _json_safe(response)


__all__ = [
    "AdapterContractError",
    "AdapterError",
    "DIMENSIONS",
    "SOURCE_REVISION",
    "run_dimension",
]

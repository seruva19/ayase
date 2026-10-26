"""Pairwise speech-naturalness evaluation with SpeechJudge-GRM.

Compares a candidate audio file (Output A) with a reference/comparator file
(Output B), conditioned on their intended transcript. Ratings are contextual
1--10 judgments rather than independent MOS estimates.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import math
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

SOURCE_REVISION = "1fbd1c883b03f3f7437506b421f7cf57bad78071"
MODEL_ID = "RMSnow/SpeechJudge-GRM"
MODEL_REVISION = "8d4ce4ddf54c5e7ac0b457fc7b561cec1bd784c1"
PROTOCOL_URL = (
    "https://raw.githubusercontent.com/AmphionTeam/SpeechJudge/"
    + SOURCE_REVISION
    + "/infer/utils.py"
)
# SHA-256 of infer/utils.py at SOURCE_REVISION. The file is data: it is parsed,
# not imported or executed.
PROTOCOL_SHA256 = "a3d35e093bb23a67f55d6300f67e5f9d5ebbe4187b1c7741ce5797a25939bcf3"
_RATING = re.compile(
    r"Output\s*A\s*:\s*(\d+(?:\.\d+)?)\s*.*?"
    r"Output\s*B\s*:\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE | re.DOTALL,
)
_VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}


def extract_ratings(response: str) -> Optional[Tuple[float, float]]:
    """Return the last finite in-range A/B rating pair in a response."""
    matches = _RATING.findall(response.replace("**", ""))
    if not matches:
        return None
    a, b = (float(value) for value in matches[-1])
    if not (math.isfinite(a) and math.isfinite(b) and 1.0 <= a <= 10.0 and 1.0 <= b <= 10.0):
        return None
    return a, b


def _loading_problems(info: Dict[str, Any]) -> Tuple[list[Any], list[Any], list[Any], list[Any]]:
    missing = list(info.get("missing_keys", []))
    mismatched = list(info.get("mismatched_keys", []))
    errors = list(info.get("error_msgs", []))
    unexpected = [
        key for key in info.get("unexpected_keys", [])
        if not (key.startswith("talker.") or key.startswith("token2wav."))
    ]
    return missing, mismatched, unexpected, errors


def _safe_protocol_value(node: ast.AST, values: Dict[str, str]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
        return node.value
    if isinstance(node, ast.List):
        return [_safe_protocol_value(item, values) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _safe_protocol_value(key, values): _safe_protocol_value(value, values)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.JoinedStr):
        pieces = []
        for item in node.values:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                pieces.append(item.value)
            elif isinstance(item, ast.FormattedValue):
                if item.conversion != -1 or item.format_spec is not None:
                    raise ValueError("unsupported formatted protocol value")
                if not isinstance(item.value, ast.Name) or item.value.id != "target_text":
                    raise ValueError("unsupported formatted protocol name")
                pieces.append(str(_safe_protocol_value(item.value, values)))
            else:
                raise ValueError("unsupported joined protocol value")
        return "".join(pieces)
    raise ValueError(f"unsupported protocol syntax: {type(node).__name__}")


def load_protocol_conversation(
    source: Path, target_text: str, audio_a: Path, audio_b: Path
) -> list[dict[str, Any]]:
    """Extract only build_cot_conversation's literal return data from pinned source."""
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("SpeechJudge protocol source SHA-256 mismatch")
    tree = ast.parse(payload.decode("utf-8"), filename=str(source))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_cot_conversation"
    ]
    if len(functions) != 1:
        raise ValueError("SpeechJudge protocol function is missing or ambiguous")
    function = functions[0]
    arguments = function.args
    if (
        [arg.arg for arg in arguments.args] != ["target_text", "wav_path_a", "wav_path_b"]
        or arguments.defaults or arguments.kw_defaults or arguments.vararg or arguments.kwarg
        or arguments.posonlyargs or arguments.kwonlyargs or function.decorator_list
    ):
        raise ValueError("SpeechJudge protocol function has unsupported arguments")
    body = function.body
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        raise ValueError("SpeechJudge protocol function has unsupported structure")
    result = _safe_protocol_value(
        body[0].value,
        {"target_text": target_text, "wav_path_a": str(audio_a), "wav_path_b": str(audio_b)},
    )
    if not isinstance(result, list) or len(result) != 2:
        raise ValueError("SpeechJudge protocol returned an invalid conversation")
    expected = [
        ("system", ["text"]),
        ("user", ["text", "text", "text", "audio", "text", "audio", "text", "text"]),
    ]
    for message, (role, types) in zip(result, expected):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("SpeechJudge protocol contains an invalid message")
        content = message["content"]
        if message["role"] != role or not isinstance(content, list) or len(content) != len(types):
            raise ValueError("SpeechJudge protocol conversation shape changed")
        for item, item_type in zip(content, types):
            value_key = "audio" if item_type == "audio" else "text"
            if not isinstance(item, dict) or set(item) != {"type", value_key}:
                raise ValueError("SpeechJudge protocol contains an invalid content item")
            if item["type"] != item_type or not isinstance(item[value_key], str):
                raise ValueError("SpeechJudge protocol content type changed")
    if result[1]["content"][3]["audio"] != str(audio_a):
        raise ValueError("SpeechJudge protocol changed Output A ordering")
    if result[1]["content"][5]["audio"] != str(audio_b):
        raise ValueError("SpeechJudge protocol changed Output B ordering")
    return result


class SpeechJudgeModule(PipelineModule):
    name = "speechjudge"
    implementation_status = "implemented"
    implementation_notes = "Pinned SpeechJudge GRM thinker checkpoint/source protocol with Qwen audio transport, native paired scoring and parsed candidate/comparator/margin/preference. Declared sampling/protocol adaptations retain the trained intended backend."
    implementation_review = "Source inspection; confidence medium; pretrained inference not validated."
    description = "SpeechJudge-GRM target-text-conditioned pairwise speech naturalness"
    input_type = "audio"
    implementation_sources = ("../third_party/speechjudge/NOTICE",)
    default_config = {
        "model_path": None,
        "protocol_path": None,
        "reference_audio_path": None,
        "target_text": None,
        "models_dir": "models",
        "device_map": "auto",
        "dtype": "bfloat16",
        "protocol": "hf",
        "max_new_tokens": 1024,
        "managed_operation_timeout": 7200,
    }
    models = [{
        "id": MODEL_ID,
        "type": "huggingface",
        "revision": MODEL_REVISION,
        "auto_download": True,
        "task": "Pairwise target-text-conditioned speech naturalness judge",
        "size": "22.36 GB (five safetensors shards)",
        "notes": "CC-BY-NC-4.0 model; non-commercial use only",
    }]
    metric_info = {
        "speechjudge_candidate_score": "Contextual Output A rating (1-10, higher=better)",
        "speechjudge_comparator_score": "Contextual Output B rating (1-10, higher=better)",
        "speechjudge_margin": "Mean Output A minus Output B rating (-9 to 9)",
        "speechjudge_preference": "Pairwise preference: -1=B, 0=tie, 1=A",
    }
    metric_groups = {field: "audio" for field in metric_info}
    provenance = {field: "adapted" for field in metric_info}
    deviations = {
        field: (
            "Uses the published native thinker-only branch with portable SDPA instead of the "
            "source default full-Omni/FlashAttention route; Ayase bounds parsed ratings and "
            "exposes numeric pairwise aggregates. scaling_10 follows the released code's mean "
            "of valid scores, not the paper's majority-vote Voting@10 protocol"
        )
        for field in metric_info
    }
    sources = {
        field: (
            "SpeechJudge-GRM source " + SOURCE_REVISION
            + " — https://github.com/AmphionTeam/SpeechJudge/tree/" + SOURCE_REVISION
        )
        for field in metric_info
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend: Optional[str] = None
        self._model = None
        self._processor = None
        self._protocol_file: Optional[Path] = None
        self._model_identity: Dict[str, Any] = {}

    def setup(self) -> None:
        # Deliberately lazy: process() can reject incomplete samples before any
        # protocol, environment, or checkpoint acquisition.
        if self.test_mode:
            return
        protocol = str(self.config.get("protocol", "hf"))
        if protocol not in {"hf", "scaling_10"}:
            self._backend = "unavailable"
            logger.warning("SpeechJudge protocol must be 'hf' or 'scaling_10'")
            return
        self._backend = "deferred"

    def _target_text(self, sample: Sample) -> Optional[str]:
        configured = self.config.get("target_text")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        if sample.caption is not None and isinstance(sample.caption.text, str):
            if sample.caption.text.strip():
                return sample.caption.text.strip()
        try:
            text = sample.path.with_suffix(".txt").read_text(encoding="utf-8").strip()
            return text or None
        except (OSError, UnicodeError):
            return None

    def _inputs(self, sample: Sample) -> Optional[Tuple[Path, Path, str]]:
        if sample.is_video:
            return None
        candidate = sample.path
        configured = self.config.get("reference_audio_path")
        comparator = Path(configured) if configured else sample.reference_path
        target = self._target_text(sample)
        if (
            comparator is None or target is None
            or candidate.suffix.lower() in _VIDEO_SUFFIXES
            or comparator.suffix.lower() in _VIDEO_SUFFIXES
            or not candidate.is_file() or not comparator.is_file()
        ):
            return None
        return candidate, comparator, target

    def _acquire_protocol(self, models_dir: Path) -> Path:
        configured = self.config.get("protocol_path")
        if configured:
            path = Path(configured).resolve()
        else:
            path = models_dir / ".ayase" / "protocols" / f"speechjudge-{SOURCE_REVISION}" / "utils.py"
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                with urllib.request.urlopen(PROTOCOL_URL, timeout=60) as response:
                    payload = response.read(131073)
                if len(payload) > 131072:
                    raise ValueError("SpeechJudge protocol source exceeds 128 KiB")
                temporary.write_bytes(payload)
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != PROTOCOL_SHA256:
                    temporary.unlink(missing_ok=True)
                    raise ValueError("downloaded SpeechJudge protocol SHA-256 mismatch")
                temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != PROTOCOL_SHA256:
            raise ValueError("SpeechJudge protocol source SHA-256 mismatch")
        return path

    def _prepare_backend(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        import torch
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        from ayase.config import download_hf_snapshot

        dtype_name = str(self.config.get("dtype", "bfloat16"))
        dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        if dtype_name not in dtypes:
            raise ValueError("dtype must be 'bfloat16', 'float16', or 'float32'")
        models_dir = Path(self.config.get("models_dir", "models")).resolve()
        self._protocol_file = self._acquire_protocol(models_dir)
        configured = self.config.get("model_path")
        model_path = configured or download_hf_snapshot(
            MODEL_ID, str(models_dir), revision=MODEL_REVISION
        )
        self._processor = Qwen2_5OmniProcessor.from_pretrained(str(model_path))
        loaded = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            str(model_path), torch_dtype=dtypes[dtype_name],
            device_map=self.config.get("device_map", "auto"),
            attn_implementation="sdpa", output_loading_info=True,
        )
        if not isinstance(loaded, tuple) or len(loaded) != 2 or not isinstance(loaded[1], dict):
            raise RuntimeError("SpeechJudge loader did not return (model, loading_info)")
        self._model, info = loaded
        missing, mismatched, unexpected, errors = _loading_problems(info)
        if missing or mismatched or unexpected or errors:
            self._model = None
            self._processor = None
            raise RuntimeError(
                "SpeechJudge checkpoint load mismatch: "
                f"missing={missing[:3]}, mismatched={mismatched[:3]}, "
                f"unexpected={unexpected[:3]}, errors={errors[:3]}"
            )
        self._model.eval()
        self._backend = "speechjudge-grm-thinker-sdpa"
        self._model_identity = {
            "configured_model_path": str(Path(configured).resolve()) if configured else None,
            "resolved_model_path": str(Path(model_path).resolve()),
            "model_override": configured is not None,
            "declared_model_id": None if configured else MODEL_ID,
            "declared_model_revision": None if configured else MODEL_REVISION,
            "dtype": dtype_name,
            "device_map": self.config.get("device_map", "auto"),
        }

    def _generate(self, conversation: list[dict[str, Any]]) -> str:
        import torch
        from qwen_omni_utils import process_mm_info

        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self._processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        )
        device = getattr(self._model, "device", next(self._model.parameters()).device)
        inputs = inputs.to(device)
        features = inputs.get("input_features")
        if features is not None and features.is_floating_point():
            thinker = getattr(self._model, "thinker", self._model)
            tower = getattr(thinker, "audio_tower", None)
            if tower is not None:
                inputs["input_features"] = features.to(dtype=next(tower.parameters()).dtype)
        prompt_length = inputs["input_ids"].shape[1]
        kwargs: Dict[str, Any] = {
            "use_audio_in_video": False, "do_sample": True,
            "max_new_tokens": int(self.config.get("max_new_tokens", 1024)),
            "eos_token_id": [151645], "pad_token_id": 151643,
        }
        if self.config.get("protocol") == "scaling_10":
            kwargs.update(temperature=1.0, top_p=1.0, top_k=50)
        with torch.inference_mode():
            generated = self._model.generate(**inputs, **kwargs)
        if isinstance(generated, tuple):
            generated = generated[0]
        suffix = generated[:, prompt_length:]
        return self._processor.batch_decode(
            suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def process(self, sample: Sample) -> Sample:
        if self.test_mode or not (
            self._backend == "deferred"
            or isinstance(self._backend, str) and self._backend.startswith("speechjudge-")
        ):
            return sample
        try:
            resolved = self._inputs(sample)
            if resolved is None:
                return sample
            candidate, comparator, target = resolved
            self._prepare_backend()
            if self._protocol_file is None:
                return sample
            conversation = load_protocol_conversation(
                self._protocol_file, target, candidate, comparator
            )
            count = 10 if self.config.get("protocol") == "scaling_10" else 1
            if count == 10:
                import torch
                torch.manual_seed(0)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)
            responses = [self._generate(conversation) for _ in range(count)]
            valid = [pair for response in responses if (pair := extract_ratings(response)) is not None]
            if not valid:
                sample.metadata["speechjudge"] = {
                    "protocol": self.config.get("protocol"), "responses": responses,
                    "valid_generations": 0, "total_generations": count,
                    "candidate": str(candidate), "comparator": str(comparator),
                    "source_revision": SOURCE_REVISION, "backend": self._backend,
                    "runtime_variant": "author_thinker_sdpa",
                    "model": dict(self._model_identity),
                }
                return sample
            score_a = sum(pair[0] for pair in valid) / len(valid)
            score_b = sum(pair[1] for pair in valid) / len(valid)
            margin = score_a - score_b
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.speechjudge_candidate_score = score_a
            sample.quality_metrics.speechjudge_comparator_score = score_b
            sample.quality_metrics.speechjudge_margin = margin
            sample.quality_metrics.speechjudge_preference = float((margin > 0) - (margin < 0))
            sample.metadata["speechjudge"] = {
                "protocol": self.config.get("protocol"), "responses": responses,
                "ratings": [{"candidate": a, "comparator": b} for a, b in valid],
                "valid_generations": len(valid), "total_generations": count,
                "candidate": str(candidate), "comparator": str(comparator),
                "target_text": target, "source_revision": SOURCE_REVISION,
                "protocol_sha256": PROTOCOL_SHA256, "backend": self._backend,
                "runtime_variant": "author_thinker_sdpa",
                "aggregation": (
                    "single" if count == 1 else "released_code_mean_of_valid_scores"
                ),
                "attention": "sdpa", "max_new_tokens": int(self.config.get("max_new_tokens", 1024)),
                "model": dict(self._model_identity),
            }
        except Exception as exc:
            if self._model is None or self._processor is None:
                self._backend = "unavailable"
            logger.warning("SpeechJudge failed for %s: %s", sample.path, exc)
        return sample

    def on_dispose(self) -> None:
        self._model = None
        self._processor = None
        self._backend = None
        self._model_identity.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        super().on_dispose()

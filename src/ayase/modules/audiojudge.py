"""Hosted audio evaluation using the complete released AudioJudge protocol.

Supports every canonical pairwise and pointwise task/prompt variant, strict JSON
verdict parsing, caller-labelled ICL, and source-compatible audio concatenation.
The default model is the documented operational successor gpt-audio-1.5; this is
an explicit transport/model adaptation, not a paper-score reproduction claim.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.third_party.audiojudge.protocol import (
    SOURCE_REVISION,
    PairwiseExample,
    PointwiseExample,
    ResponseContract,
    build_gemini_messages,
    build_openai_messages,
    canonical_contract,
    custom_contract,
    normalize_wav,
    parse_response,
)

logger = logging.getLogger(__name__)

OUTPUT_FIELDS = {
    "audiojudge_score": "AudioJudge pointwise ordinal rating (1-5; not calibrated MOS)",
    "audiojudge_preference": "Pairwise candidate result (-1 loss, 0 tie, +1 win)",
    "audiojudge_match": "Pairwise pronunciation/speaker match verdict (0 or 1)",
    "audiojudge_choice": "Descriptive pairwise choice (1 or 2; currently speaking speed)",
}
_CONTEXT_TASKS = {"speakbench", "speakbench508", "chatbotarena", "chatbotarena_BA"}


class AudioJudgeModule(PipelineModule):
    name = "audiojudge"
    description = "Generic AudioJudge hosted audio evaluator"
    input_type = "audio"
    implementation_sources = (
        "../third_party/audiojudge/protocol.py",
        "../third_party/audiojudge/system_prompts.py",
        "../third_party/audiojudge/system_prompts_pointwise.py",
        "../third_party/audiojudge/LICENSE",
        "../third_party/audiojudge/NOTICE",
    )
    default_config = {
        "provider": "openai",
        "model": "gpt-audio-1.5",
        "api_base": None,
        "api_key": None,
        "openai_api_key": None,
        "gemini_api_key": None,
        "organization": None,
        "project": None,
        "timeout_s": 900,
        "temperature": 1e-8,
        "max_tokens": 800,
        "mode": "pairwise",
        "task": "somos",
        "variant": "standard_cot",
        "concatenation_method": None,
        "comparator_path": None,
        "instruction_path": None,
        "instruction_text": None,
        "user_prompt": None,
        "examples": [],
        "system_prompt": None,
        "custom_semantic": None,
        "custom_schema": None,
        "models_dir": "models",
    }
    models = [
        {
            "id": "gpt-audio-1.5",
            "type": "other",
            "auto_download": False,
            "url": "https://developers.openai.com/api/docs/models/gpt-audio-1.5",
            "task": "Default hosted AudioJudge audio/text evaluation transport",
            "notes": "Operational successor adaptation; API access and billing are user-provided, and historical AudioJudge numbers are not reproduced by this model substitution.",
        },
        {
            "id": "gpt-4o-audio-preview",
            "type": "other",
            "auto_download": False,
            "url": "https://developers.openai.com/api/docs/deprecations",
            "task": "Historical released AudioJudge default",
            "notes": "Retired 2026-05-07; retained only as provenance and never selected by default.",
        },
        {
            "id": "gemini-3.8-flash",
            "type": "other",
            "auto_download": False,
            "url": "https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash",
            "task": "Optional caller-selected Gemini audio-understanding transport",
            "notes": "Requires explicit provider/model configuration; compatibility is not a numerical-equivalence claim.",
        },
    ]
    metric_info = OUTPUT_FIELDS
    metric_groups = {field: "audio" for field in OUTPUT_FIELDS}
    provenance = {field: "adapted" for field in OUTPUT_FIELDS}
    sources = {
        field: (
            "AudioJudge source "
            + SOURCE_REVISION
            + " — https://github.com/Woodygan/AudioJudge/tree/"
            + SOURCE_REVISION
        )
        for field in OUTPUT_FIELDS
    }
    deviations = {
        field: (
            "Preserves the pinned literal prompts, ICL/message ordering, boundary assets, and "
            "concatenation timing; uses gpt-audio-1.5 instead of the retired released preview, "
            "modern provider SDKs, managed full-duration media decoding, and a mono PCM16 "
            "heterogeneous-input portability repair. The Gemini adaptation explicitly applies "
            "temperature, output-token, and JSON MIME controls that the released branch ignored. "
            "No numerical reproduction is claimed"
        )
        for field in OUTPUT_FIELDS
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend: Optional[str] = None
        self._client: Any = None

    def setup(self) -> None:
        if self.test_mode:
            return
        provider = str(self.config.get("provider", "openai")).strip().lower()
        if provider not in {"openai", "gemini"}:
            logger.warning("AudioJudge provider must be openai or gemini")
            self._backend = "unavailable"
            return
        key = self._api_key(provider)
        if not key:
            logger.warning("AudioJudge %s credentials are unavailable", provider)
            self._backend = "unavailable"
            return
        try:
            if provider == "openai":
                from openai import OpenAI

                kwargs: Dict[str, Any] = {
                    "api_key": key,
                    "timeout": float(self.config.get("timeout_s", 900)),
                }
                if self.config.get("api_base"):
                    kwargs["base_url"] = self.config["api_base"]
                if self.config.get("organization"):
                    kwargs["organization"] = self.config["organization"]
                if self.config.get("project"):
                    kwargs["project"] = self.config["project"]
                self._client = OpenAI(**kwargs)
                self._backend = "openai-chat-completions-adapted"
            else:
                if self._model_name(provider) == "gpt-audio-1.5":
                    raise ValueError("Gemini provider requires an explicit Gemini model override")
                from google import genai

                http_options: Dict[str, Any] = {
                    "timeout": int(float(self.config.get("timeout_s", 900)) * 1000)
                }
                if self.config.get("api_base"):
                    http_options["base_url"] = self.config["api_base"]
                kwargs = {"api_key": key, "http_options": http_options}
                self._client = genai.Client(**kwargs)
                self._backend = "gemini-generate-content-adapted"
        except Exception as exc:
            self._client = None
            self._backend = "unavailable"
            logger.warning("AudioJudge setup failed: %s", exc)

    def _api_key(self, provider: str) -> Optional[str]:
        scoped = self.config.get(provider + "_api_key")
        if isinstance(scoped, str) and scoped:
            return scoped
        configured = self.config.get("api_key")
        if isinstance(configured, str) and configured:
            return configured
        names = {
            "openai": ("OPENAI_API_KEY",),
            "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        }
        for name in names[provider]:
            value = os.environ.get(name)
            if value:
                return value
        return None

    def _model_name(self, provider: str) -> str:
        value = self.config.get("model")
        if not isinstance(value, str) or not value.strip():
            if provider == "openai":
                return "gpt-audio-1.5"
            raise ValueError("Gemini provider requires an explicit model")
        return value.strip()

    def _contract(self) -> ResponseContract:
        mode = str(self.config.get("mode", "pairwise")).strip().lower()
        prompt = self.config.get("system_prompt")
        if prompt is None:
            return canonical_contract(
                mode,
                str(self.config.get("task", "somos")),
                str(self.config.get("variant", "standard_cot")),
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("custom system_prompt must be a nonempty string")
        semantic = self.config.get("custom_semantic")
        schema = self.config.get("custom_schema")
        if not isinstance(semantic, str) or not isinstance(schema, dict):
            raise ValueError("custom system_prompt requires custom_semantic and custom_schema")
        return custom_contract(mode, prompt, semantic, schema)

    def _path_value(self, sample: Sample, name: str) -> Optional[Path]:
        value = sample.metadata.get("audiojudge_" + name, self.config.get(name))
        if value is None:
            return None
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"{name} must be a nonempty filesystem path")
        return Path(value).expanduser().resolve()

    def _materialize_audio(self, source: Path, output: Path) -> Path:
        if not source.is_file():
            raise FileNotFoundError(f"audio input does not exist: {source}")
        try:
            with wave.open(str(source), "rb") as stream:
                pcm = stream.getcomptype() == "NONE"
            if pcm:
                return normalize_wav(source, output)
        except (EOFError, OSError, ValueError, wave.Error):
            pass
        from ayase.managed_ffmpeg import resolve_media_tool

        ffmpeg = resolve_media_tool(
            "ffmpeg", models_dir=self.config.get("models_dir", "models")
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(ffmpeg),
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output),
            ],
            check=True,
            capture_output=True,
            timeout=int(self.config.get("timeout_s", 900)),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return normalize_wav(output, output.with_name(output.stem + "_normalized.wav"))

    @staticmethod
    def _example_rows(sample: Sample, config: Dict[str, Any]) -> Sequence[Any]:
        rows = sample.metadata.get("audiojudge_examples", config.get("examples", []))
        if not isinstance(rows, list):
            raise ValueError("AudioJudge examples must be a list")
        if len(rows) > 10:
            raise ValueError("AudioJudge supports at most 10 caller-labelled examples")
        return rows

    def _examples(
        self,
        sample: Sample,
        contract: ResponseContract,
        work: Path,
        instruction_mode: bool,
    ) -> Sequence[Any]:
        parsed: List[Any] = []
        for index, row in enumerate(self._example_rows(sample, self.config), 1):
            if not isinstance(row, dict) or not isinstance(row.get("output"), str):
                raise ValueError(f"AudioJudge example {index} needs a literal output string")
            parse_response(row["output"], contract)
            if contract.mode == "pointwise":
                if not isinstance(row.get("audio_path"), (str, Path)):
                    raise ValueError(f"pointwise example {index} needs audio_path")
                audio = self._materialize_audio(
                    Path(row["audio_path"]).expanduser().resolve(),
                    work / f"example_{index}_audio.wav",
                )
                parsed.append(PointwiseExample(audio, row["output"]))
                continue
            for key in ("audio1_path", "audio2_path"):
                if not isinstance(row.get(key), (str, Path)):
                    raise ValueError(f"pairwise example {index} needs {key}")
            audio1 = self._materialize_audio(
                Path(row["audio1_path"]).expanduser().resolve(),
                work / f"example_{index}_audio1.wav",
            )
            audio2 = self._materialize_audio(
                Path(row["audio2_path"]).expanduser().resolve(),
                work / f"example_{index}_audio2.wav",
            )
            instruction = None
            if instruction_mode:
                value = row.get("instruction_path")
                if not isinstance(value, (str, Path)) or not str(value).strip():
                    raise ValueError(
                        f"instruction-mode example {index} needs its own instruction_path"
                    )
                instruction = self._materialize_audio(
                    Path(value).expanduser().resolve(),
                    work / f"example_{index}_instruction.wav",
                )
            parsed.append(PairwiseExample(audio1, audio2, row["output"], instruction))
        return parsed

    def _user_prompt(self, contract: ResponseContract) -> Tuple[Optional[str], str]:
        configured = self.config.get("user_prompt")
        if configured is not None and (
            not isinstance(configured, str) or not configured.strip()
        ):
            raise ValueError("user_prompt must be a nonempty string when configured")
        instruction = self.config.get("instruction_text")
        if instruction is not None and (
            not isinstance(instruction, str) or not instruction.strip()
        ):
            raise ValueError("instruction_text must be a nonempty string when configured")
        if instruction:
            default = (
                "Please provide your response according to this audio clip:"
                if contract.mode == "pointwise"
                else "Please provide your response according to these audio clips:"
            )
            prompt = (configured.strip() if configured else default) + (
                "\n\nOriginal instruction (explicit textual context): " + instruction.strip()
            )
            return prompt, "text"
        return configured.strip() if configured else None, "none"

    def _validate_instruction_contract(
        self,
        contract: ResponseContract,
        instruction_path: Optional[Path],
        instruction_kind: str,
        example_count: int,
    ) -> None:
        if contract.mode == "pointwise" and instruction_path is not None:
            raise ValueError("released pointwise AudioJudge has no instruction audio input")
        needs_context = contract.task in _CONTEXT_TASKS and not (
            contract.mode == "pairwise" and contract.variant == "speech_quality_cot"
        )
        if needs_context and instruction_path is None and instruction_kind != "text":
            raise ValueError(
                f"{contract.mode}/{contract.task}/{contract.variant} requires the original instruction context"
            )
        if contract.mode == "pointwise" and contract.task in {
            "speakbench",
            "speakbench508",
        } and instruction_kind != "text":
            raise ValueError("pointwise SpeakBench requires explicit instruction_text")
        if example_count and instruction_kind == "text" and instruction_path is None:
            raise ValueError(
                "text-only instruction augmentation does not support ICL; use instruction_path per example"
            )

    def _call_openai(
        self, messages: Sequence[Dict[str, Any]], model: str
    ) -> Tuple[str, Dict[str, Any]]:
        response = self._client.chat.completions.create(
            model=model,
            messages=list(messages),
            modalities=["text"],
            temperature=float(self.config.get("temperature", 1e-8)),
            max_completion_tokens=int(self.config.get("max_tokens", 800)),
        )
        text = response.choices[0].message.content
        if not isinstance(text, str) or not text:
            raise ValueError("OpenAI returned no text response")
        usage = getattr(response, "usage", None)
        return text, {
            "actual_model": getattr(response, "model", None),
            "request_id": getattr(response, "_request_id", None),
            "response_id": getattr(response, "id", None),
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
        }

    def _call_gemini(
        self, messages: Sequence[Any], model: str
    ) -> Tuple[str, Dict[str, Any]]:
        from google.genai import types

        contents = [
            types.Part.from_bytes(data=item["data"], mime_type=item["mime_type"])
            if isinstance(item, dict)
            and set(item) == {"mime_type", "data"}
            and isinstance(item["data"], bytes)
            else item
            for item in messages
        ]
        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=float(self.config.get("temperature", 1e-8)),
                max_output_tokens=int(self.config.get("max_tokens", 800)),
                response_mime_type="application/json",
            ),
        )
        text = response.text
        if not isinstance(text, str) or not text:
            raise ValueError("Gemini returned no text response")
        usage = getattr(response, "usage_metadata", None)
        return text, {
            "actual_model": getattr(response, "model_version", None),
            "response_id": getattr(response, "response_id", None),
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
        }

    def _set_metric(self, sample: Sample, semantic: str, value: float) -> None:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        field = {
            "score": "audiojudge_score",
            "preference": "audiojudge_preference",
            "match": "audiojudge_match",
            "choice": "audiojudge_choice",
        }[semantic]
        setattr(sample.quality_metrics, field, value)

    def process(self, sample: Sample) -> Sample:
        if (
            self.test_mode
            or not self._mounted
            or self._backend in {None, "unavailable"}
            or self._client is None
        ):
            return sample
        raw_response: Optional[str] = None
        contract: Optional[ResponseContract] = None
        request_meta: Dict[str, Any] = {}
        response_meta: Dict[str, Any] = {}
        try:
            provider = str(self.config.get("provider", "openai")).strip().lower()
            model = self._model_name(provider)
            contract = self._contract()
            method = self.config.get("concatenation_method")
            if method is None:
                method = (
                    "examples_concatenation"
                    if contract.mode == "pointwise"
                    else "examples_and_test_concatenation"
                )
            if not isinstance(method, str):
                raise ValueError("concatenation_method must be a string")
            user_prompt, instruction_kind = self._user_prompt(contract)
            comparator_source = self._path_value(sample, "comparator_path")
            instruction_source = self._path_value(sample, "instruction_path")
            rows = self._example_rows(sample, self.config)
            self._validate_instruction_contract(
                contract, instruction_source, instruction_kind, len(rows)
            )

            with tempfile.TemporaryDirectory(prefix="ayase-audiojudge-") as temporary:
                work = Path(temporary)
                candidate = self._materialize_audio(sample.path.resolve(), work / "candidate.wav")
                comparator = (
                    self._materialize_audio(comparator_source, work / "comparator.wav")
                    if comparator_source is not None
                    else None
                )
                if contract.mode == "pairwise" and comparator is None:
                    raise ValueError("pairwise AudioJudge requires comparator_path")
                instruction = (
                    self._materialize_audio(instruction_source, work / "instruction.wav")
                    if instruction_source is not None
                    else None
                )
                examples = self._examples(
                    sample, contract, work, instruction_mode=instruction is not None
                )
                builder = build_openai_messages if provider == "openai" else build_gemini_messages
                messages = builder(
                    mode=contract.mode,
                    candidate_path=candidate,
                    comparator_path=comparator,
                    instruction_path=instruction,
                    examples=examples,
                    system_prompt=contract.system_prompt,
                    user_prompt=user_prompt,
                    concatenation_method=method,
                    work_dir=work,
                    models_dir=self.config.get("models_dir", "models"),
                )
                request_meta = {
                    "provider": provider,
                    "requested_model": model,
                    "transport": self._backend,
                    "temperature": float(self.config.get("temperature", 1e-8)),
                    "max_tokens": int(self.config.get("max_tokens", 800)),
                    "concatenation_method": method,
                    "num_examples": len(examples),
                    "instruction_context": "audio" if instruction is not None else instruction_kind,
                    "message_items": len(messages),
                }
                if provider == "openai":
                    raw_response, response_meta = self._call_openai(messages, model)
                else:
                    raw_response, response_meta = self._call_gemini(messages, model)
            mapped, parsed = parse_response(raw_response, contract)
            self._set_metric(sample, contract.semantic, mapped)
            sample.metadata[self.name] = {
                "success": True,
                "raw_response": raw_response,
                "parsed_response": parsed,
                "mode": contract.mode,
                "task": contract.task,
                "variant": contract.variant,
                "semantic": contract.semantic,
                "schema": {
                    "exact_keys": sorted(contract.exact_keys),
                    "result_key": contract.result_key,
                    "result_type": contract.result_type,
                    "allowed": list(contract.allowed),
                },
                "request": request_meta,
                "response": response_meta,
                "source_revision": SOURCE_REVISION,
                "historical_default_model": "gpt-4o-audio-preview",
                "model_migration": "gpt-4o-audio-preview -> gpt-audio-1.5"
                if provider == "openai" and model == "gpt-audio-1.5"
                else None,
                "numerical_reproduction_claimed": False,
                "pcm_policy": "full-duration mono signed PCM16; modal concat rate, ties first",
            }
        except Exception as exc:
            sample.metadata[self.name] = {
                "success": False,
                "error": str(exc),
                "raw_response": raw_response,
                "mode": contract.mode if contract is not None else None,
                "task": contract.task if contract is not None else None,
                "variant": contract.variant if contract is not None else None,
                "semantic": contract.semantic if contract is not None else None,
                "request": request_meta,
                "response": response_meta,
                "source_revision": SOURCE_REVISION,
            }
            logger.warning("AudioJudge failed for %s: %s", sample.path, exc)
        return sample

    def on_dispose(self) -> None:
        try:
            close = getattr(self._client, "close", None)
            if not callable(close):
                api_client = getattr(self._client, "_api_client", None)
                transport = getattr(api_client, "_httpx_client", None)
                close = getattr(transport, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            logger.warning("AudioJudge client cleanup failed: %s", exc)
        finally:
            self._client = None
            self._backend = None
            super().on_dispose()

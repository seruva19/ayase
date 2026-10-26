"""Scientific experiment video evaluation using the released VideoScience-Judge protocol.

Scores five source-defined dimensions from 1 to 4 (higher is better) with a
weighted 1--4 overall score. Requires authoritative expected-phenomenon input;
this module never infers scientific ground truth from the candidate video.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.third_party.videoscience.protocol import (
    CHECKLIST_KEYS,
    DATASET_REVISION,
    SOURCE_REVISION,
    build_checklist_prompt,
    build_judge_prompt,
    compute_overall,
    data_url,
    extract_frames,
    parse_checklist,
    parse_judgment,
)

logger = logging.getLogger(__name__)

DATASET_ID = "lmgame/VideoScienceBench"
DATASET_FILE = "data_filtered.jsonl"
DATASET_URL = (
    "https://huggingface.co/datasets/lmgame/VideoScienceBench/resolve/"
    + DATASET_REVISION
    + "/"
    + DATASET_FILE
)
DATASET_SHA256 = "c015f689cdf9c537601b2ab5ec19c9ebaf9c67b0bd3b5e3cad0a9351f07455cb"
OUTPUT_FIELDS = {
    "videoscience_score": "Weighted VideoScience-Judge overall score (1-4, higher=better)",
    "videoscience_prompt_score": "Prompt consistency score (1-4, higher=better)",
    "videoscience_phenomenon_score": "Expected phenomenon score (1-4, higher=better)",
    "videoscience_coherence_score": "Spatio-temporal coherence score (1-4, higher=better)",
    "videoscience_immutability_score": "Object immutability score (1-4, higher=better)",
    "videoscience_dynamism_score": "Other physical-law dynamism score (1-4, higher=better)",
}


class VideoScienceJudgeModule(PipelineModule):
    name = "videoscience_judge"
    implementation_status = "implemented"
    implementation_notes = "Released base/checklist VLM judge has real provider calls, source rubrics/weights and strict judgment parsing; unreleased salient-frame and optional CV stage are excluded. Full provider runtime not tested."
    implementation_review = "Source inspection; confidence medium; pretrained inference not validated."
    description = "VideoScience VLM judge for generated scientific experiment videos"
    input_type = "video"
    implementation_sources = (
        "../third_party/videoscience/protocol.py",
        "../third_party/videoscience/LICENSE",
        "../third_party/videoscience/NOTICE",
    )
    default_config = {
        "provider": "openai",
        "model": None,
        "api_base": None,
        "api_key": None,
        "openai_api_key": None,
        "gemini_api_key": None,
        "anthropic_api_key": None,
        "timeout_s": 900,
        "max_output_tokens": 20480,
        "experiment_prompt": None,
        "expected_phenomenon": None,
        "scenario_id": None,
        "models_dir": "models",
        "canonical_prompt_policy": "override",
        "reference_video_path": None,
        "checklist": None,
        "generate_checklist": False,
        "checklist_model": "models/gemini-2.5-flash",
        "source_text": None,
        "source_url": None,
        "cv_evidence": None,
        "gemini_poll_s": 1.0,
    }
    models = [
        {
            "id": "gpt-5-pro",
            "type": "other",
            "auto_download": False,
            "url": "https://platform.openai.com/docs/models/gpt-5-pro",
            "task": "Default OpenAI Responses VLM judge",
            "notes": "Released judge model; OpenAI has announced shutdown on 2026-12-11. Configure a compatible replacement explicitly when needed.",
        },
        {
            "id": "models/gemini-2.5-flash",
            "type": "other",
            "auto_download": False,
            "url": "https://ai.google.dev/gemini-api/docs/models",
            "task": "Optional checklist generation",
            "notes": "Google limits Gemini 2.5 access to prior active users; no shutdown date was announced as of 2026-10-25.",
        },
        {
            "id": "gemini-2.5-pro",
            "type": "other",
            "auto_download": False,
            "url": "https://ai.google.dev/gemini-api/docs/models",
            "task": "Google native-video VLM judge",
            "notes": "Google limits Gemini 2.5 access to prior active users; no shutdown date was announced as of 2026-10-25.",
        },
        {
            "id": "claude-sonnet-4-5",
            "type": "other",
            "auto_download": False,
            "url": "https://docs.anthropic.com/en/docs/about-claude/models/overview",
            "task": "Anthropic frame-based VLM judge",
            "notes": "Historical provider alias; Anthropic has announced retirement on 2026-11-30. Configure a compatible replacement explicitly when needed.",
        },
    ]
    metric_info = OUTPUT_FIELDS
    metric_groups = {
        "videoscience_score": "alignment",
        "videoscience_prompt_score": "alignment",
        "videoscience_phenomenon_score": "alignment",
        "videoscience_coherence_score": "temporal",
        "videoscience_immutability_score": "temporal",
        "videoscience_dynamism_score": "motion",
    }
    provenance = {key: "adapted" for key in OUTPUT_FIELDS}
    deviations = {
        key: (
            "Uses the released base/checklist VLM judge and source scoring weights, with strict "
            "finite 1-4 parsing and explicit canonical/general input provenance; it does not "
            "implement the unreleased salient-frame stage or optional CV pipeline"
        )
        for key in OUTPUT_FIELDS
    }
    sources = {
        key: "VideoScience source "
        + SOURCE_REVISION
        + " — https://github.com/hao-ai-lab/VideoScience/tree/"
        + SOURCE_REVISION
        for key in OUTPUT_FIELDS
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend: Optional[str] = None
        self._session: Any = requests
        self._gemini_client: Any = None

    def setup(self) -> None:
        if self.test_mode:
            return
        provider = str(self.config.get("provider", "openai")).lower()
        if provider not in {"openai", "gemini", "anthropic"}:
            logger.warning("VideoScience provider must be openai, gemini, or anthropic")
            self._backend = "unavailable"
            return
        key = self._api_key(provider)
        if not key:
            logger.warning("VideoScience %s credentials are unavailable", provider)
            self._backend = "unavailable"
            return
        if provider == "gemini" or self.config.get("generate_checklist"):
            gemini_key = self._api_key("gemini")
            if not gemini_key:
                logger.warning("VideoScience Gemini checklist credentials are unavailable")
                self._backend = "unavailable"
                return
            try:
                from google import genai

                self._gemini_client = genai.Client(api_key=gemini_key)
            except Exception as exc:
                logger.warning("VideoScience Gemini client unavailable: %s", type(exc).__name__)
                self._backend = "unavailable"
                return
        self._backend = provider + "-released-protocol"

    def _api_key(self, provider: str) -> Optional[str]:
        scoped = self.config.get(provider + "_api_key")
        if isinstance(scoped, str) and scoped:
            return scoped
        configured_provider = str(self.config.get("provider", "openai")).lower()
        configured = self.config.get("api_key")
        if provider == configured_provider and isinstance(configured, str) and configured:
            return configured
        names = {
            "openai": ("OPENAI_API_KEY",),
            "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            "anthropic": ("ANTHROPIC_API_KEY",),
        }
        for name in names[provider]:
            value = os.environ.get(name)
            if value:
                return value
        return None

    def _model_name(self, provider: str) -> str:
        configured = self.config.get("model")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        return {
            "openai": "gpt-5-pro",
            "gemini": "gemini-2.5-pro",
            "anthropic": "claude-sonnet-4-5",
        }[provider]

    def _canonical_file(self) -> Path:
        root = Path(self.config.get("models_dir", "models")).resolve()
        return root / "videoscience" / DATASET_REVISION / DATASET_FILE

    def _load_scenario(self, scenario_id: str) -> Dict[str, Any]:
        path = self._canonical_file()
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            response = self._session.get(
                DATASET_URL, timeout=int(self.config.get("timeout_s", 900))
            )
            response.raise_for_status()
            payload = response.content
            if hashlib.sha256(payload).hexdigest() != DATASET_SHA256:
                raise ValueError("canonical VideoScience dataset SHA-256 mismatch")
            temporary.write_bytes(payload)
            temporary.replace(path)
        found = None
        count = 0
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if not isinstance(row, dict) or not all(
                    isinstance(row.get(key), str) and row[key].strip()
                    for key in ("vid", "prompt", "expected phenomenon")
                ):
                    raise ValueError("canonical VideoScience JSONL has invalid schema")
                count += 1
                if row["vid"] == scenario_id:
                    if found is not None:
                        raise ValueError("canonical scenario id is not unique")
                    found = row
        if count != 160:
            raise ValueError(f"canonical VideoScience dataset has {count} rows, expected 160")
        if found is None:
            raise ValueError(f"canonical VideoScience scenario not found: {scenario_id}")
        return found

    def _inputs(self, sample: Sample) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        configured_prompt = self.config.get("experiment_prompt")
        caption = (
            sample.caption.text.strip() if sample.caption and sample.caption.text.strip() else None
        )
        prompt = (
            configured_prompt.strip()
            if isinstance(configured_prompt, str) and configured_prompt.strip()
            else caption
        )
        configured_truth = self.config.get("expected_phenomenon")
        metadata_truth = sample.metadata.get("expected_phenomenon")
        truth = (
            configured_truth
            if isinstance(configured_truth, str) and configured_truth.strip()
            else metadata_truth
        )
        scenario = self.config.get("scenario_id")
        provenance: Dict[str, Any] = {
            "mode": "general",
            "scenario_id": None,
            "caption_mismatch": False,
        }
        if scenario is not None:
            row = self._load_scenario(str(scenario))
            canonical_prompt = row["prompt"].strip()
            mismatch = prompt is not None and prompt != canonical_prompt
            policy = self.config.get("canonical_prompt_policy", "override")
            if mismatch and policy == "reject":
                raise ValueError("supplied prompt does not match canonical scenario")
            if policy not in {"override", "reject"}:
                raise ValueError("canonical_prompt_policy must be override or reject")
            prompt = canonical_prompt
            truth = row["expected phenomenon"].strip()
            provenance.update(
                mode="canonical-160",
                scenario_id=row["vid"],
                caption_mismatch=mismatch,
                supplied_prompt=configured_prompt or caption,
                canonical_prompt=canonical_prompt,
                dataset_id=DATASET_ID,
                dataset_revision=DATASET_REVISION,
            )
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(truth, str)
            or not truth.strip()
        ):
            return None
        return prompt.strip(), truth.strip(), provenance

    def _configured_checklist(self) -> Optional[Dict[str, List[str]]]:
        value = self.config.get("checklist")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("checklist must be an object")
        return parse_checklist(json.dumps(value))

    def _generate_checklist(
        self, prompt: str, truth: str
    ) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
        source_text = self.config.get("source_text")
        source_url = self.config.get("source_url")
        youtube = isinstance(source_url, str) and (
            "youtube.com" in source_url or "youtu.be" in source_url
        )
        if not youtube and not (
            isinstance(source_text, str)
            and source_text.strip()
            and isinstance(source_url, str)
            and source_url.strip()
        ):
            raise ValueError(
                "checklist generation requires authoritative source_text and source_url, "
                "or a YouTube source_url"
            )
        if self._gemini_client is None:
            raise ValueError("released checklist generation requires a ready Gemini client")
        reference = (
            "Reference is a YouTube video containing the expected phenomenon."
            if youtube
            else "Reference Source Text:\n" + source_text.strip()
        )
        checklist_prompt = build_checklist_prompt(prompt, truth, reference, source_url or "")
        contents: List[Any] = [checklist_prompt]
        if youtube:
            from google.genai import types

            contents.append(types.Part.from_uri(file_uri=source_url, mime_type="video/*"))
        schema = {
            "type": "object",
            "properties": {
                key: {"type": "array", "items": {"type": "string"}} for key in CHECKLIST_KEYS
            },
            "required": list(CHECKLIST_KEYS),
        }
        from google.genai import types

        response = self._gemini_client.models.generate_content(
            model=str(self.config.get("checklist_model")),
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_json_schema=schema
            ),
        )
        checklist = parse_checklist(response.text or "")
        return checklist, {
            "kind": "generated",
            "source_url": source_url,
            "source_text_provided": bool(source_text),
            "model": self.config.get("checklist_model"),
        }

    @staticmethod
    def _frame_content(
        prompt: str,
        candidate: List[Tuple[float, bytes]],
        reference: List[Tuple[float, bytes]],
        style: str,
    ) -> List[Dict[str, Any]]:
        image_type, text_type = (
            ("input_image", "input_text") if style == "openai" else ("image", "text")
        )
        content: List[Dict[str, Any]] = [{"type": text_type, "text": prompt}]
        for timestamp, payload in candidate[:20]:
            image = (
                {"type": image_type, "image_url": data_url(payload)}
                if style == "openai"
                else {
                    "type": image_type,
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": data_url(payload).split(",", 1)[1],
                    },
                }
            )
            content.extend(
                [image, {"type": text_type, "text": f"Candidate frame t={timestamp:.1f}s"}]
            )
        if reference:
            content.append({"type": text_type, "text": "Reference video frames:"})
            for timestamp, payload in reference[:10]:
                image = (
                    {"type": image_type, "image_url": data_url(payload)}
                    if style == "openai"
                    else {
                        "type": image_type,
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": data_url(payload).split(",", 1)[1],
                        },
                    }
                )
                content.extend(
                    [image, {"type": text_type, "text": f"Reference frame t={timestamp:.1f}s"}]
                )
        return content

    def _http_json(
        self, url: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        response = self._session.post(
            url, headers=headers, json=payload, timeout=int(self.config.get("timeout_s", 900))
        )
        response.raise_for_status()
        data = response.json()
        return data, {"request_id": response.headers.get("x-request-id", "")}

    def _call_openai(
        self, prompt: str, sample: Sample, reference_path: Optional[Path]
    ) -> Tuple[str, Dict[str, Any]]:
        candidate = extract_frames(sample.path, 24)
        reference = extract_frames(reference_path, 12) if reference_path else []
        content = self._frame_content(prompt, candidate, reference, "openai")
        payload = {
            "model": self._model_name("openai"),
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": int(self.config.get("max_output_tokens", 20480)),
        }
        base = self.config.get("api_base") or os.environ.get(
            "OPENAI_API_BASE", "https://api.openai.com"
        )
        raw, ids = self._http_json(
            base.rstrip("/") + "/v1/responses",
            {
                "Authorization": "Bearer " + self._api_key("openai"),
                "Content-Type": "application/json",
            },
            payload,
        )
        text = raw.get("output_text")
        if not text:
            text = "\n".join(
                part.get("text", "")
                for item in raw.get("output", [])
                for part in item.get("content", [])
                if isinstance(part, dict)
            )
        return text or "", {
            "raw": raw,
            "actual_model": raw.get("model"),
            **ids,
            "candidate_timestamps": [t for t, _ in candidate[:20]],
            "reference_timestamps": [t for t, _ in reference[:10]],
        }

    def _call_anthropic(
        self, prompt: str, sample: Sample, reference_path: Optional[Path]
    ) -> Tuple[str, Dict[str, Any]]:
        candidate = extract_frames(sample.path, 24)
        reference = extract_frames(reference_path, 12) if reference_path else []
        payload = {
            "model": self._model_name("anthropic"),
            "max_tokens": int(self.config.get("max_output_tokens", 20480)),
            "messages": [
                {
                    "role": "user",
                    "content": self._frame_content(prompt, candidate, reference, "anthropic"),
                }
            ],
        }
        base = self.config.get("api_base") or "https://api.anthropic.com"
        headers = {
            "x-api-key": self._api_key("anthropic"),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        raw, ids = self._http_json(base.rstrip("/") + "/v1/messages", headers, payload)
        text = "\n".join(
            item.get("text", "") for item in raw.get("content", []) if item.get("type") == "text"
        )
        return text, {
            "raw": raw,
            "actual_model": raw.get("model"),
            **ids,
            "candidate_timestamps": [t for t, _ in candidate[:20]],
            "reference_timestamps": [t for t, _ in reference[:10]],
        }

    def _wait_file(self, file_obj: Any) -> Any:
        deadline = time.monotonic() + int(self.config.get("timeout_s", 900))
        while time.monotonic() < deadline:
            current = self._gemini_client.files.get(name=file_obj.name)
            state = getattr(current.state, "name", current.state)
            if state == "ACTIVE":
                return current
            if state in {"FAILED", "DELETED"}:
                raise RuntimeError(f"Gemini upload entered {state} state")
            time.sleep(float(self.config.get("gemini_poll_s", 1.0)))
        raise TimeoutError("Gemini upload did not become ACTIVE")

    def _call_gemini(
        self, prompt: str, sample: Sample, reference_path: Optional[Path]
    ) -> Tuple[str, Dict[str, Any]]:
        from google.genai import types

        uploaded = []
        try:
            candidate = self._gemini_client.files.upload(file=str(sample.path))
            uploaded.append(candidate)
            candidate = self._wait_file(candidate)
            contents: List[Any] = [prompt, candidate]
            if reference_path:
                reference = self._gemini_client.files.upload(file=str(reference_path))
                uploaded.append(reference)
                reference = self._wait_file(reference)
                contents.extend(["Reference video follows.", reference])
            response = self._gemini_client.models.generate_content(
                model=self._model_name("gemini"),
                contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            raw = response.to_dict() if hasattr(response, "to_dict") else {"text": response.text}
            actual_model = getattr(response, "model_version", None)
            request_id = getattr(response, "response_id", None)
            return response.text or "", {
                "raw": raw,
                "actual_model": actual_model,
                "request_id": request_id,
                "candidate_timestamps": ["uploaded:file"],
                "reference_timestamps": ["uploaded:file"] if reference_path else [],
            }
        finally:
            for file_obj in uploaded:
                try:
                    self._gemini_client.files.delete(name=file_obj.name)
                except Exception as exc:
                    logger.warning(
                        "VideoScience Gemini upload cleanup failed: %s", type(exc).__name__
                    )

    def process(self, sample: Sample) -> Sample:
        if (
            self.test_mode
            or not self._mounted
            or not self._backend
            or self._backend == "unavailable"
            or not sample.is_video
        ):
            return sample
        try:
            inputs = self._inputs(sample)
            if inputs is None or not sample.path.is_file() or sample.path.stat().st_size == 0:
                return sample
            experiment_prompt, truth, input_provenance = inputs
            reference_value = self.config.get("reference_video_path")
            reference_path = Path(reference_value) if reference_value else sample.reference_path
            if reference_path is not None and (
                not reference_path.is_file() or reference_path.stat().st_size == 0
            ):
                return sample
            checklist = self._configured_checklist()
            checklist_provenance: Dict[str, Any] = {"kind": "none"}
            if checklist:
                checklist_provenance = {"kind": "caller-provided"}
            elif self.config.get("generate_checklist"):
                checklist, checklist_provenance = self._generate_checklist(experiment_prompt, truth)
            cv_evidence = self.config.get("cv_evidence")
            if cv_evidence is not None and not isinstance(cv_evidence, dict):
                raise ValueError("cv_evidence must be an object")
            prompt = build_judge_prompt(truth, experiment_prompt, checklist, cv_evidence)
            provider = str(self.config.get("provider")).lower()
            callers = {
                "openai": self._call_openai,
                "anthropic": self._call_anthropic,
                "gemini": self._call_gemini,
            }
            output_text, details = callers[provider](prompt, sample, reference_path)
            judgment = parse_judgment(output_text)
            scores = judgment["scores"]
            values = {
                "videoscience_score": compute_overall(scores),
                "videoscience_prompt_score": scores["prompt_consistency"],
                "videoscience_phenomenon_score": scores["expected_phenomenon"],
                "videoscience_coherence_score": scores["coherence"],
                "videoscience_immutability_score": scores["immutability"],
                "videoscience_dynamism_score": scores["dynamism"],
            }
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            for field, value in values.items():
                setattr(sample.quality_metrics, field, value)
            sample.metadata["videoscience"] = {
                "provider": provider,
                "requested_model": self._model_name(provider),
                "actual_model": details.get("actual_model"),
                "request_id": details.get("request_id"),
                "source_revision": SOURCE_REVISION,
                "dataset_revision": DATASET_REVISION,
                "runtime_variant": "released-base" if not checklist else "released-checklist",
                "inputs": input_provenance,
                "experiment_prompt": experiment_prompt,
                "expected_phenomenon": truth,
                "checklist": checklist,
                "checklist_provenance": checklist_provenance,
                "cv_evidence_provided": cv_evidence is not None,
                "scores": dict(scores),
                "overall_weights": {
                    "prompt_consistency": 0.2,
                    "expected_phenomenon": 0.3,
                    "coherence": 0.2,
                    "immutability": 0.15,
                    "dynamism": 0.15,
                },
                "explanations": judgment.get("explanations"),
                "evidence": judgment.get("evidence"),
                "candidate_evidence": details.get("candidate_timestamps"),
                "reference_evidence": details.get("reference_timestamps"),
                "response_text": output_text,
                "raw_response": details.get("raw"),
            }
        except Exception as exc:
            logger.warning("VideoScience-Judge failed for %s: %s", sample.path, type(exc).__name__)
        return sample

    def on_dispose(self) -> None:
        if self._gemini_client is not None:
            close = getattr(self._gemini_client, "close", None)
            if callable(close):
                close()
            else:
                api_client = getattr(self._gemini_client, "_api_client", None)
                http_client = getattr(api_client, "_httpx_client", None)
                close = getattr(http_client, "close", None)
                if callable(close):
                    close()
        self._gemini_client = None
        self._backend = None
        super().on_dispose()

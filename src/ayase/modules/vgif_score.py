"""VGIF-Score objective QA and prompt-specific AutoRubric video evaluation.

Runs the released dependency-round Gemini protocol against explicit VGIF
annotations. Higher normalized scores are better. Arbitrary caption-to-rubric
generation is not part of the released method and is intentionally unsupported.
"""

import base64
import json
import logging
import math
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.third_party.vgif_score.protocol import (
    DIMENSIONS,
    build_autorubric_prompt,
    build_dependency_round_prompt,
    build_dependency_rounds,
    evaluate_predictions,
    validate_annotations,
)

logger = logging.getLogger(__name__)


class VGIFScoreModule(PipelineModule):
    name = "vgif_score"
    implementation_status = "implemented"
    implementation_notes = "Concrete API-only published protocol with explicit annotations, dependency rounds/gated objective accuracy and four-dimension rubric mean; requires configured compatible endpoint/API key. Protocol helper inspection needed, no paid call run."
    implementation_review = "Source inspection; confidence medium; pretrained inference not validated."
    input_type = "video"
    description = "VGIF-Score via dependency-gated Gemini video QA and prompt AutoRubric"
    default_config = {
        "base_url": "",
        "objective_model": "gemini-3.1-pro-preview",
        "rubric_model": "gemini-3-pro-preview",
        "timeout": 180,
        "retries": 2,
        "format_retries": 2,
        "annotations": None,
    }
    models = [
        {"id": "gemini-3.1-pro-preview", "type": "other", "task": "VGIF objective video QA judge", "url": "https://ai.google.dev/gemini-api/docs/models", "auto_download": False, "notes": "Remote preview alias; returned modelVersion is recorded per call."},
        {"id": "gemini-3-pro-preview", "type": "other", "task": "VGIF AutoRubric video judge", "url": "https://ai.google.dev/gemini-api/docs/models", "auto_download": False, "notes": "Remote preview alias; returned modelVersion is recorded per call."},
    ]
    metric_info = {
        "vgif_score": "VGIF overall score: equal mean of objective and subjective scores (0-1, higher=better)",
        "vgif_objective_score": "Dependency-gated QA accuracy (0-1, higher=better)",
        "vgif_subjective_score": "Mean of four raw AutoRubric ratings divided by 5 (0.2-1, higher=better)",
        "vgif_cinematography_score": "Raw prompt-specific cinematography rating (1-5, higher=better)",
        "vgif_purity_score": "Raw prompt-specific visual-purity rating (1-5, higher=better)",
        "vgif_motion_score": "Raw prompt-specific motion-smoothness rating (1-5, higher=better)",
        "vgif_physics_score": "Raw prompt-specific physics-adherence rating (1-5, higher=better)",
    }
    metric_groups = {key: "alignment" for key in metric_info}
    provenance = "published"
    deviations = {
        field: "Malformed annotation graphs are rejected before upload; boolean rubric ratings and extra rubric dimensions are rejected instead of coerced."
        for field in metric_info
    }
    implementation_sources = (
        "../third_party/vgif_score/protocol.py",
        "../third_party/vgif_score/LICENSE",
        "../third_party/vgif_score/NOTICE",
    )
    sources = {
        key: "VGIF-Score — https://github.com/PRIS-CV/VGIF-SCORE/tree/b7558fe27bb13b97cd6039dfa421da4d11c4f2bc"
        for key in metric_info
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.base_url = str(self.config.get("base_url") or os.getenv("VGIF_BASE_URL") or "").rstrip("/")
        self.objective_model = str(self.config.get("objective_model", self.default_config["objective_model"]))
        self.rubric_model = str(self.config.get("rubric_model", self.default_config["rubric_model"]))
        self.timeout = int(self.config.get("timeout", 180))
        self.retries = int(self.config.get("retries", 2))
        self.format_retries = int(self.config.get("format_retries", 2))
        self._api_key: Optional[str] = None
        self._session = None
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        api_key = self.config.get("api_key") or os.getenv("VGIF_API_KEY")
        if not api_key:
            logger.warning("VGIF-Score unavailable: VGIF_API_KEY is not configured")
            self._backend = "unavailable"
            return
        if not self.base_url:
            logger.warning("VGIF-Score unavailable: VGIF_BASE_URL is not configured")
            self._backend = "unavailable"
            return
        try:
            import requests

            self._session = requests.Session()
            self._session.trust_env = False
            self._session.headers.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
            self._api_key = str(api_key)
            self._backend = "gemini_api"
        except ImportError:
            logger.warning("VGIF-Score unavailable: requests is not installed")
            self._backend = "unavailable"
        except Exception as exc:
            logger.warning("VGIF-Score setup failed: %s", self._safe_error(exc))
            self._session = None
            self._api_key = None
            self._backend = "unavailable"

    def on_dispose(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session = None
        self._api_key = None
        self._backend = None
        super().on_dispose()

    def process(self, sample: Sample) -> Sample:
        if self._backend != "gemini_api" or not sample.is_video:
            return sample
        try:
            annotations = self._annotations(sample)
            validate_annotations(annotations)
            if not sample.path.is_file():
                raise ValueError("VGIF video path does not exist")

            video_bytes = sample.path.read_bytes()
            if not video_bytes:
                raise ValueError("VGIF video is empty")
            mime_type = mimetypes.guess_type(sample.path.name)[0] or "video/mp4"
            encoded = base64.b64encode(video_bytes).decode("ascii")

            answers: dict[str, dict[str, str]] = {}
            calls = []
            rounds = build_dependency_rounds(annotations["original_qa_pairs"])
            for index, round_qas in enumerate(rounds, start=1):
                prompt = build_dependency_round_prompt(
                    round_index=index,
                    total_rounds=len(rounds),
                    current_round_qas=round_qas,
                    locked_answers=answers,
                )
                payload, call = self._judge(self.objective_model, prompt, mime_type, encoded)
                round_answers = self._normalize_answers(payload, round_qas)
                answers.update(round_answers)
                call["round"] = index
                calls.append(call)

            evaluated, raw_correct, correct, blocked = evaluate_predictions(
                annotations["original_qa_pairs"], answers
            )
            rubric, rubric_call = self._judge(
                self.rubric_model,
                build_autorubric_prompt(annotations),
                mime_type,
                encoded,
                validator=self._normalize_rubric,
            )
            calls.append(rubric_call)

            objective = correct / len(evaluated)
            raw_ratings = {code: rubric[code]["score"] for code, _, _ in DIMENSIONS}
            subjective = sum(raw_ratings.values()) / (5.0 * len(raw_ratings))
            overall = 0.5 * objective + 0.5 * subjective
            values = [overall, objective, subjective, *raw_ratings.values()]
            if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
                raise ValueError("VGIF produced a non-finite score")

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            metrics = sample.quality_metrics
            metrics.vgif_score = overall
            metrics.vgif_objective_score = objective
            metrics.vgif_subjective_score = subjective
            metrics.vgif_cinematography_score = float(raw_ratings["Cin"])
            metrics.vgif_purity_score = float(raw_ratings["Pur"])
            metrics.vgif_motion_score = float(raw_ratings["Mot"])
            metrics.vgif_physics_score = float(raw_ratings["Phy"])
            sample.metadata["vgif_score"] = {
                "protocol": "dependency_rounds",
                "source_commit": "b7558fe27bb13b97cd6039dfa421da4d11c4f2bc",
                "prompt": annotations["prompt"],
                "questions": evaluated,
                "raw_correct_count": raw_correct,
                "gated_correct_count": correct,
                "dependency_blocked_count": blocked,
                "answer_coverage": sum(row["predicted_answer"] != "UNKNOWN" for row in evaluated) / len(evaluated),
                "rubric": rubric,
                "requested_models": {
                    "objective": self.objective_model,
                    "rubric": self.rubric_model,
                },
                "calls": calls,
                "annotation_provenance": annotations.get("provenance"),
            }
        except Exception as exc:
            logger.warning("VGIF-Score evaluation failed: %s", self._safe_error(exc))
        return sample

    def _annotations(self, sample: Sample) -> dict[str, Any]:
        if "vgif_annotations" in sample.metadata:
            annotations = sample.metadata["vgif_annotations"]
        else:
            annotations = self.config.get("annotations")
        if not isinstance(annotations, dict):
            raise ValueError("explicit sample.metadata.vgif_annotations or config annotations are required")
        merged = dict(annotations)
        if not isinstance(merged.get("prompt"), str) or not merged["prompt"].strip():
            if sample.caption and sample.caption.text.strip():
                merged["prompt"] = sample.caption.text.strip()
        if "dimensions" in merged:
            return merged
        autorubric = annotations.get("autorubric")
        if isinstance(autorubric, dict):
            merged["dimensions"] = autorubric.get("dimensions")
            return merged
        return merged

    def _judge(self, model: str, prompt: str, mime_type: str, encoded: str, validator=None):
        last_error: Optional[Exception] = None
        for format_attempt in range(1, self.format_retries + 2):
            try:
                response = self._post(model, self._request_payload(model, prompt, mime_type, encoded))
                response.raise_for_status()
                envelope = response.json()
                raw_text = self._extract_text(envelope)
                result = self._extract_json(raw_text)
                validated = validator(result) if validator is not None else result
                return validated, {
                    "requested_model": model,
                    "model_version": envelope.get("modelVersion"),
                    "response_id": envelope.get("responseId"),
                    "format_attempt": format_attempt,
                    "raw_response": raw_text,
                }
            except Exception as exc:
                last_error = exc
        raise ValueError(f"invalid judge response after format retries: {self._safe_error(last_error)}")

    def _post(self, model: str, payload: dict[str, Any]):
        import requests

        endpoint = f"{self.base_url}/v1beta/models/{model}:generateContent"
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                return self._session.post(endpoint, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 * (attempt + 1), 5))
        raise last_error

    @staticmethod
    def _request_payload(model: str, prompt: str, mime_type: str, encoded: str) -> dict[str, Any]:
        return {
            "model": model,
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": encoded}},
            ]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
        }

    @staticmethod
    def _extract_text(envelope: dict[str, Any]) -> str:
        for candidate in envelope.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if isinstance(part.get("text"), str) and part["text"].strip():
                    return part["text"]
        raise ValueError("judge response contained no text")

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                cleaned = cleaned[start : end + 1]
        value = json.loads(cleaned)
        if not isinstance(value, dict):
            raise ValueError("judge response must be an object")
        return value

    @staticmethod
    def _normalize_answers(payload: dict[str, Any], qas: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
        items = payload.get("answers", [])
        if not isinstance(items, list):
            return {}
        result = {}
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            cleaned = re.sub(r"[^a-z]", "", str(item.get("answer", "")).strip().lower())
            if cleaned in {"yes", "y", "true"}:
                answer = "Yes"
            elif cleaned in {"no", "n", "false"}:
                answer = "No"
            else:
                answer = "UNKNOWN"
            result[item["id"]] = {"answer": answer, "reason": str(item.get("reason", "")).strip()}
        return result

    @staticmethod
    def _normalize_rubric(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        scores = payload.get("scores")
        if not isinstance(scores, dict) or set(scores) != {code for code, _, _ in DIMENSIONS}:
            raise ValueError("rubric must contain exactly Cin, Pur, Mot, and Phy")
        result = {}
        for code, _, _ in DIMENSIONS:
            item = scores[code]
            if not isinstance(item, dict) or isinstance(item.get("score"), bool):
                raise ValueError(f"invalid rubric score for {code}")
            value = item.get("score")
            if isinstance(value, int):
                score = value
            elif isinstance(value, float) and value.is_integer():
                score = int(value)
            elif isinstance(value, str):
                match = re.search(r"\b[1-5]\b", value)
                score = int(match.group(0)) if match else None
            else:
                score = None
            if score is None or not 1 <= score <= 5:
                raise ValueError(f"rubric score for {code} must be an integer 1..5")
            result[code] = {"score": score, "reason": str(item.get("reason", "")).strip()}
        return result

    def _safe_error(self, exc: Optional[Exception]) -> str:
        text = str(exc or "unknown error")
        if self._api_key:
            text = text.replace(self._api_key, "<redacted>")
        return text[:500]

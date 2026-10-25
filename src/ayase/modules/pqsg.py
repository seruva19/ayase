"""Prompt-derived question-scene-graph evaluation of video semantics and physics.

Uses the released PQSG seven-example question-generation prompt and two-call
batched video QA protocol. Scores range from 0 to 1; higher is better.
Effective defaults are Gemini 2.5 Flash at 24 fps, or GPT-5 at 2 fps when the
OpenAI backend is selected. Credentials use GOOGLE_API_KEY or OPENAI_API_KEY;
explicit config keys and expert OpenAI-compatible base URLs are also supported.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.third_party.pqsg import protocol

logger = logging.getLogger(__name__)


class PQSGModule(PipelineModule):
    name = "pqsg"
    description = "Prompt-derived question graph verification with dependency-aware scoring"
    input_type = "video"
    requires_video = True
    default_config = {
        "prompt": None,
        "qg_backend": "gemini",
        "qg_model": None,
        "qg_api_key": None,
        "qg_base_url": None,
        "qa_backend": "gemini",
        "qa_model": None,
        "qa_fps": None,
        "qa_api_key": None,
        "qa_base_url": None,
    }
    metric_groups = {
        "pqsg_score": "alignment",
        "pqsg_raw_score": "alignment",
        "pqsg_object_score": "alignment",
        "pqsg_action_score": "alignment",
        "pqsg_physics_score": "alignment",
    }
    metric_info = {
        "pqsg_score": "Dependency-propagated PQSG score (0-1, higher=better)",
        "pqsg_raw_score": "Raw yes-answer ratio before dependency propagation (0-1)",
        "pqsg_object_score": "Conditional object score among ancestor-eligible nodes",
        "pqsg_action_score": "Conditional action score among ancestor-eligible nodes",
        "pqsg_physics_score": "Conditional physics score among ancestor-eligible nodes",
    }
    _source = (
        "PQSG released source at immutable revision 03afdb0e6f9ed68d363a68eea5ca4fd90d53848e — "
        "https://github.com/atinpothiraj/pqsg/tree/03afdb0e6f9ed68d363a68eea5ca4fd90d53848e"
    )
    sources = dict.fromkeys(metric_info, _source)
    _deviation = (
        "Ayase follows the released seven-example, two-batched-call implementation rather than "
        "the paper's one-example, individual-question prose; rejects malformed, dangling, cyclic, "
        "backward, and O-to-O graphs before video upload; and reads prompts/sidecars as UTF-8. "
        "Provider model and QA FPS remain configurable."
    )
    deviations = dict.fromkeys(metric_info, _deviation)
    implementation_sources = (
        "../third_party/pqsg/protocol.py",
        "../third_party/pqsg/qg_prompt.txt",
        "../third_party/pqsg/LICENSE",
        "../third_party/pqsg/NOTICE",
    )
    models = [
        {
            "id": "gemini-2.5-flash",
            "type": "other",
            "url": "https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash",
            "task": "Hosted PQSG graph generation and video QA; configurable OpenAI route also supported",
            "auto_download": False,
            "notes": "Released default. Google access may be limited to projects that previously used Gemini 2.5; configure a currently available compatible model when needed.",
        },
        {
            "id": "gpt-5",
            "type": "other",
            "url": "https://platform.openai.com/docs/models/gpt-5",
            "task": "Hosted PQSG graph generation and frame-based video QA",
            "auto_download": False,
        },
    ]
    provenance = {field: "adapted" for field in metric_info}

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = None
        self._qg_client = None
        self._qa_client = None
        self._owned_clients = []

    def _model(self, role: str, backend: str) -> str:
        key = f"{role}_model"
        configured = self.config.get(key)
        if configured is not None:
            if not isinstance(configured, str) or not configured.strip():
                raise ValueError(f"PQSG {key} must be a non-empty string")
            return configured.strip()
        return "gpt-5" if backend == "openai" else "gemini-2.5-flash"

    def _qa_fps(self, backend: str) -> int:
        configured = self.config.get("qa_fps")
        if configured is not None:
            fps = int(configured)
            if fps <= 0:
                raise ValueError("PQSG qa_fps must be positive")
            return fps
        return 2 if backend == "openai" else 24

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            clients = {}
            for role in ("qg", "qa"):
                backend = str(self.config.get(f"{role}_backend", "gemini"))
                if backend not in {"gemini", "openai"}:
                    raise ValueError(f"Unsupported PQSG {role} backend: {backend}")
                self._model(role, backend)
                if role == "qa":
                    self._qa_fps(backend)
                configured = self.config.get(f"{role}_client")
                if configured is not None:
                    clients[role] = configured
                    continue
                key = self.config.get(f"{role}_api_key") or os.environ.get(
                    "GOOGLE_API_KEY" if backend == "gemini" else "OPENAI_API_KEY"
                )
                if not key:
                    raise RuntimeError(f"missing credentials for PQSG {role} {backend} provider")
                reuse = next(
                    (
                        clients[other]
                        for other in clients
                        if self.config.get(f"{other}_backend", "gemini") == backend
                        and self.config.get(f"{other}_api_key")
                        == self.config.get(f"{role}_api_key")
                        and self.config.get(f"{other}_base_url")
                        == self.config.get(f"{role}_base_url")
                        and self.config.get(f"{other}_client") is None
                    ),
                    None,
                )
                if reuse is not None:
                    clients[role] = reuse
                elif backend == "gemini":
                    from google import genai

                    clients[role] = genai.Client(api_key=key)
                    self._owned_clients.append(clients[role])
                elif backend == "openai":
                    from openai import OpenAI

                    kwargs = {"api_key": key}
                    base_url = self.config.get(f"{role}_base_url")
                    if base_url:
                        kwargs["base_url"] = base_url
                    clients[role] = OpenAI(**kwargs)
                    self._owned_clients.append(clients[role])
            self._qg_client, self._qa_client = clients["qg"], clients["qa"]
            self._backend = "api"
        except Exception as exc:
            for client in dict.fromkeys(self._owned_clients):
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            self._owned_clients.clear()
            self._backend = "unavailable"
            self._qg_client = self._qa_client = None
            logger.warning("PQSG setup failed: %s", self._redact(str(exc)))

    def _redact(self, message: str) -> str:
        for key in (
            self.config.get("qg_api_key"),
            self.config.get("qa_api_key"),
            os.environ.get("GOOGLE_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
        ):
            if key:
                message = message.replace(str(key), "[REDACTED]")
        return message

    def _redact_trace(self, value):
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, list):
            return [self._redact_trace(item) for item in value]
        if isinstance(value, dict):
            return {key: self._redact_trace(item) for key, item in value.items()}
        return value

    def _prompt(self, sample: Sample) -> Optional[str]:
        explicit = self.config.get("prompt")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        if sample.caption and sample.caption.text.strip():
            return sample.caption.text.strip()
        sidecar = sample.path.with_suffix(".txt")
        if sidecar.is_file():
            return sidecar.read_text(encoding="utf-8").strip() or None
        return None

    def process(self, sample: Sample) -> Sample:
        if self.test_mode or self._backend != "api" or not sample.is_video:
            return sample
        try:
            if not sample.path.is_file() or sample.path.stat().st_size <= 0:
                return sample
            prompt = self._prompt(sample)
            if not prompt:
                return sample
            graph, qg_trace = protocol.generate_psg(
                prompt,
                backend=self.config.get("qg_backend", "gemini"),
                model=self._model("qg", self.config.get("qg_backend", "gemini")),
                api_key=self.config.get("qg_api_key"),
                base_url=self.config.get("qg_base_url"),
                client=self._qg_client,
            )
            protocol.validate_graph(graph)
            qa_backend = self.config.get("qa_backend", "gemini")
            common = dict(
                video_path=str(sample.path),
                graph=graph,
                model=self._model("qa", qa_backend),
                fps=self._qa_fps(qa_backend),
                api_key=self.config.get("qa_api_key"),
                client=self._qa_client,
            )
            if qa_backend == "gemini":
                answers, qa_trace = protocol.answer_gemini(**common)
            elif qa_backend == "openai":
                answers, qa_trace = protocol.answer_openai(
                    **common, base_url=self.config.get("qa_base_url")
                )
            else:
                raise ValueError(f"Unsupported PQSG QA backend: {qa_backend}")
            correct = protocol._correct(answers)
            raw = sum(correct.values()) / len(correct)
            score = protocol.tree_score(answers, graph["edges"])
            categories = protocol.category_scores(answers, graph["edges"])
            eligibility = protocol.eligibility_mask(answers, graph["edges"])
            values = {
                "pqsg_score": score,
                "pqsg_raw_score": raw,
                "pqsg_object_score": categories["object"],
                "pqsg_action_score": categories["action"],
                "pqsg_physics_score": categories["physics"],
            }
            metrics = sample.quality_metrics or QualityMetrics()
            sample.quality_metrics = metrics.model_copy(update=values)
            sample.metadata["pqsg"] = {
                "protocol": "released-batched-seven-example",
                "qg_backend": self.config.get("qg_backend", "gemini"),
                "qa_backend": qa_backend,
                "qg": self._redact_trace(qg_trace),
                "qa": self._redact_trace(qa_trace),
                "graph": graph,
                "answers": answers,
                "conditional_category_scores": categories,
                "eligibility": eligibility,
                "coverage": qa_trace.get("coverage", 1.0),
                "question_count": len(correct),
            }
        except Exception as exc:
            logger.warning("PQSG failed for %s: %s", sample.path, self._redact(str(exc)))
        return sample

    def on_dispose(self) -> None:
        for client in dict.fromkeys(self._owned_clients):
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self._owned_clients.clear()
        self._qg_client = self._qa_client = None
        self._backend = None
        super().on_dispose()

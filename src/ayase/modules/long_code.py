"""Approximate long-context video evaluation adapted from Long-CODE.

Implements Dynamic Structure Alignment with an order-sensitive X-CLIP video
encoder and an optional three-stage hosted vision-judge workflow. The released
paper's human reference bank and author prompts are unavailable, so semantic
and combined scores are explicitly uncalibrated approximations.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Sequence

from ayase.config import download_hf_snapshot
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

PAPER_URL = "https://arxiv.org/abs/2604.17428"
ENCODER_ID = "microsoft/xclip-base-patch32"
ENCODER_REVISION = "a2e27a78a2b5d802e894b8a1ef14f3a8ce490963"
XCLIP_FRAME_COUNT = 8
ENCODER_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "pytorch_model.bin",
    "model.safetensors",
]
OUTPUT_FIELDS = (
    "long_code_approx_dsa",
    "long_code_approx_narrative_score",
    "long_code_approx_causality_score",
    "long_code_approx_identity_score",
    "long_code_approx_semantic_score",
    "long_code_approx_score",
)


def _average_ranks(values: Sequence[float]) -> Optional[list[float]]:
    """Return one-based average ranks, or ``None`` for invalid values."""
    if len(values) < 2 or any(isinstance(value, bool) for value in values):
        return None
    numbers = [float(value) for value in values]
    if any(not math.isfinite(value) for value in numbers):
        return None
    order = sorted(range(len(numbers)), key=numbers.__getitem__)
    ranks = [0.0] * len(numbers)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and numbers[order[end]] == numbers[order[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for index in order[start:end]:
            ranks[index] = average
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    """Spearman correlation with average ranks and explicit degeneracy handling."""
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if left_ranks is None or right_ranks is None:
        return None
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    left_delta = [value - left_mean for value in left_ranks]
    right_delta = [value - right_mean for value in right_ranks]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0.0:
        return None
    result = sum(a * b for a, b in zip(left_delta, right_delta)) / denominator
    return float(max(-1.0, min(1.0, result))) if math.isfinite(result) else None


class LongCodeModule(PipelineModule):
    """Long-CODE-inspired structural and semantic long-video evaluator."""

    name = "long_code"
    description = "Approximate Long-CODE long-context structural and semantic evaluation"
    input_type = "video"
    implementation_status = "approximate"
    implementation_notes = (
        "Equation-inspired DSA with X-CLIP plus an original generic three-stage hosted-judge "
        "rubric; the unreleased human reference bank and author prompts are omitted, so no "
        "official Long-CODE parity or calibration is claimed. The standard X-CLIP revision is "
        "pinned; expert encoder_model overrides must also provide their matching revision."
    )
    default_config = {
        "encoder_model": ENCODER_ID,
        "encoder_revision": ENCODER_REVISION,
        "models_dir": "models",
        "device": "auto",
        "num_frames": XCLIP_FRAME_COUNT,
        "shot_prompts": None,
        "alpha": 0.5,
        "semantic_rounds": 1,
        "judge_model": None,
        "api_key": None,
        "base_url": None,
        "timeout": 180.0,
        "max_tokens": 1000,
    }
    models = [
        {
            "id": ENCODER_ID,
            "type": "huggingface",
            "task": "Order-sensitive joint text-video encoder for approximate DSA",
            "auto_download": True,
            "notes": f"Default revision pinned to {ENCODER_REVISION}; eight-frame checkpoint.",
        },
        {
            "id": "configurable-openai-compatible-vision-judge",
            "type": "other",
            "url": "https://developers.openai.com/api/docs/guides/images-vision",
            "task": "Optional hosted caption, reasoning, and semantic scoring stages",
            "auto_download": False,
            "notes": "Requires explicit judge_model and API credentials; no request is made during setup.",
        },
    ]
    metric_info = {
        "long_code_approx_dsa": "Approximate Dynamic Structure Alignment (-1 to 1, higher=better)",
        "long_code_approx_narrative_score": "Approximate narrative coherence (0-100)",
        "long_code_approx_causality_score": "Approximate causal consistency (0-100)",
        "long_code_approx_identity_score": "Approximate cross-shot identity persistence (0-100)",
        "long_code_approx_semantic_score": "Mean of three approximate semantic dimensions (0-100)",
        "long_code_approx_score": "Ayase alpha-weighted approximate combined score (0-100)",
    }
    metric_groups = {field: "alignment" for field in metric_info}
    provenance = {field: "adapted" for field in metric_info}
    sources = {field: f"Long-CODE paper — {PAPER_URL}" for field in metric_info}
    deviations = {
        field: (
            "Uses microsoft/xclip-base-patch32 with its processor for full-video temporal "
            "encoding; represents the global story as the normalized mean of all independently "
            "encoded shot prompts so X-CLIP's text limit cannot discard later shots; uses original "
            "generic API rubrics, no human-annotated reference bank, and an Ayase-defined "
            "configurable alpha (default 0.5); no official parity is claimed"
        )
        for field in metric_info
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._processor: Any = None
        self._encoder: Any = None
        self._client: Any = None
        self._device = "cpu"
        self._encoder_backend: Optional[str] = None
        self._semantic_backend: Optional[str] = None
        self._backend: Optional[str] = None

    def setup(self) -> None:
        """Load X-CLIP and optionally construct a hosted client without API calls."""
        if self.test_mode or self._validated_config() is None:
            return
        try:
            import torch
            from transformers import XCLIPModel, XCLIPProcessor

            model_id = str(self.config.get("encoder_model", ENCODER_ID))
            revision = self.config.get("encoder_revision")
            cached = download_hf_snapshot(
                model_id,
                str(self.config.get("models_dir", "models")),
                revision=revision if isinstance(revision, str) and revision else None,
                allow_patterns=ENCODER_ALLOW_PATTERNS,
            )
            processor = XCLIPProcessor.from_pretrained(str(cached))
            encoder = XCLIPModel.from_pretrained(str(cached))
            vision_config = getattr(getattr(encoder, "config", None), "vision_config", None)
            checkpoint_frames = getattr(vision_config, "num_frames", None)
            if checkpoint_frames != XCLIP_FRAME_COUNT:
                raise ValueError(
                    "Long-CODE X-CLIP checkpoint must declare exactly "
                    f"{XCLIP_FRAME_COUNT} frames, got {checkpoint_frames!r}"
                )
            requested = str(self.config.get("device", "auto"))
            self._device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
            if self._device == "auto":
                self._device = "cpu"
            self._processor = processor
            self._encoder = encoder.to(self._device).eval()
            self._encoder_backend = "xclip-full-video"
        except Exception as exc:
            logger.warning("Long-CODE X-CLIP encoder unavailable: %s", exc)

        model = self.config.get("judge_model")
        key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not isinstance(model, str) or not model.strip() or not isinstance(key, str) or not key:
            return
        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {
                "api_key": key,
                "timeout": float(self.config.get("timeout", 180.0)),
            }
            if self.config.get("base_url"):
                kwargs["base_url"] = self.config["base_url"]
            self._client = OpenAI(**kwargs)
            self._semantic_backend = "openai-compatible-three-stage"
        except Exception as exc:
            logger.warning("Long-CODE semantic client unavailable: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if self.test_mode:
            return sample
        self._clear_outputs(sample)
        self._backend = None
        try:
            validated = self._validated_inputs(sample)
            if validated is None or self._encoder is None or self._processor is None:
                return sample
            prompts, alpha, rounds, num_frames = validated
            frames, timestamps = self._read_frames(Path(sample.path), num_frames)
            if len(frames) != num_frames or len(timestamps) != num_frames:
                return sample
            dsa, truncation = self._compute_dsa(prompts, frames)
            if dsa is None:
                return sample

            self._write_metrics(sample, {"long_code_approx_dsa": dsa})
            self._backend = self._encoder_backend or "xclip-full-video"
            status = "dsa_only"
            semantic: Optional[dict[str, float]] = None
            semantic_error_type: Optional[str] = None
            if self._client is not None:
                try:
                    semantic = self._semantic_scores(prompts, frames, timestamps, rounds)
                except Exception as exc:
                    semantic_error_type = type(exc).__name__
                    logger.warning(
                        "Long-CODE semantic evaluation failed for %s: %s", sample.path, exc
                    )
                if semantic is not None:
                    mean = sum(semantic.values()) / 3.0
                    combined = alpha * ((dsa + 1.0) * 50.0) + (1.0 - alpha) * mean
                    values = {
                        "long_code_approx_narrative_score": semantic["narrative"],
                        "long_code_approx_causality_score": semantic["causality"],
                        "long_code_approx_identity_score": semantic["identity"],
                        "long_code_approx_semantic_score": mean,
                        "long_code_approx_score": combined,
                    }
                    if self._valid_scores(values.values()):
                        self._write_metrics(sample, values)
                        status = "full_approximate"
                        self._backend += "+" + (
                            self._semantic_backend or "openai-compatible-three-stage"
                        )
                    else:
                        semantic = None
                if semantic is None:
                    status = "semantic_failed"

            sample.metadata["long_code"] = {
                "implementation_status": self.implementation_status,
                "availability": status,
                "encoder_backend": self._encoder_backend or "injected",
                "encoder_model": str(self.config.get("encoder_model", ENCODER_ID)),
                "encoder_revision": self.config.get("encoder_revision"),
                "encoder_cache": "ayase-models-dir-huggingface-snapshot",
                "semantic_backend": self._semantic_backend if self._client is not None else None,
                "semantic_error_type": semantic_error_type,
                "judge_model": self.config.get("judge_model") if self._client is not None else None,
                "shot_prompt_count": len(prompts),
                "sampled_frame_count": len(frames),
                "semantic_rounds": rounds if self._client is not None else 0,
                "alpha": alpha,
                "paper": PAPER_URL,
                "provenance": "adapted",
                "reference_bank": "omitted-unreleased",
                "official_parity": False,
                "text_truncation": truncation,
            }
        except Exception as exc:
            logger.warning("Long-CODE approximate evaluation failed for %s: %s", sample.path, exc)
        return sample

    def _validated_inputs(
        self, sample: Sample
    ) -> Optional[tuple[list[str], float, int, int]]:
        if not sample.is_video or not Path(sample.path).is_file():
            return None
        raw_prompts = self.config.get("shot_prompts")
        if raw_prompts is None:
            raw_prompts = sample.metadata.get("shot_prompts")
        if not isinstance(raw_prompts, (list, tuple)) or len(raw_prompts) < 3:
            return None
        if any(not isinstance(item, str) or not item.strip() for item in raw_prompts):
            return None
        prompts = [item.strip() for item in raw_prompts]
        validated = self._validated_config()
        if validated is None:
            return None
        alpha, rounds, num_frames = validated
        return prompts, alpha, rounds, num_frames

    def _validated_config(self) -> Optional[tuple[float, int, int]]:
        encoder_model = self.config.get("encoder_model", ENCODER_ID)
        encoder_revision = self.config.get("encoder_revision", ENCODER_REVISION)
        alpha = self.config.get("alpha", 0.5)
        rounds = self.config.get("semantic_rounds", 1)
        num_frames = self.config.get("num_frames", XCLIP_FRAME_COUNT)
        timeout = self.config.get("timeout", 180.0)
        max_tokens = self.config.get("max_tokens", 1000)
        if (
            not isinstance(encoder_model, str)
            or not encoder_model.strip()
            or not isinstance(encoder_revision, str)
            or not encoder_revision.strip()
            or isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or not 0.0 <= float(alpha) <= 1.0
            or isinstance(rounds, bool)
            or not isinstance(rounds, int)
            or rounds <= 0
            or isinstance(num_frames, bool)
            or not isinstance(num_frames, int)
            or num_frames != XCLIP_FRAME_COUNT
            or isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0.0
            or isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            return None
        return float(alpha), rounds, num_frames

    @staticmethod
    def _read_frames(path: Path, count: int) -> tuple[list[Any], list[float]]:
        import cv2
        import numpy as np

        capture = cv2.VideoCapture(str(path))
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if total < 1 or not math.isfinite(fps) or fps <= 0.0:
                return [], []
            # The released checkpoint is configured for eight frames. Repeated indices
            # preserve that shape for short clips while retaining chronological order.
            indices = np.linspace(0, total - 1, count, dtype=int)
            frames, timestamps = [], []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    return [], []
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamps.append(float(index) / fps)
            return frames, timestamps
        finally:
            capture.release()

    def _compute_dsa(
        self, prompts: Sequence[str], frames: Sequence[Any]
    ) -> tuple[Optional[float], dict[str, Any]]:
        import torch
        import torch.nn.functional as functional

        max_length = self._model_text_max_length()
        truncation = self._text_truncation(list(prompts), max_length)
        with torch.inference_mode():
            text_inputs = self._processor(
                text=list(prompts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            text_inputs = {key: value.to(self._device) for key, value in text_inputs.items()}
            text = self._encoder.get_text_features(**text_inputs)
            # transformers 4.57.x drops ``videos=`` in XCLIPProcessor.__call__ and
            # returns an empty BatchFeature. Its native image processor preserves
            # X-CLIP's resize/normalize contract and produces [1, 8, 3, H, W].
            image_processor = getattr(self._processor, "image_processor", None)
            if not callable(image_processor):
                raise ValueError("X-CLIP processor has no callable image_processor")
            video_batch = image_processor(list(frames), return_tensors="pt")
            pixel_values = video_batch.get("pixel_values")
            if (
                not isinstance(pixel_values, torch.Tensor)
                or pixel_values.ndim != 5
                or pixel_values.shape[0] != 1
                or pixel_values.shape[1] != XCLIP_FRAME_COUNT
            ):
                shape = getattr(pixel_values, "shape", None)
                raise ValueError(
                    "X-CLIP image processor must return pixel_values shaped "
                    f"[1, {XCLIP_FRAME_COUNT}, 3, H, W], got {shape!r}"
                )
            video_inputs = {"pixel_values": pixel_values.to(self._device)}
            # transformers 4.57.x has a get_video_features return_dict propagation bug.
            # The model's full native forward correctly returns the same projected,
            # temporally integrated video embedding as ``video_embeds``.
            forward = self._encoder(**text_inputs, **video_inputs, return_dict=True)
            video = forward.video_embeds
            text = functional.normalize(text.float(), dim=-1).cpu()
            video = functional.normalize(video.float(), dim=-1).cpu()[0]
        shot_text = text
        global_text = functional.normalize(shot_text.mean(dim=0), dim=0)
        text_similarities = (shot_text @ global_text).tolist()
        video_similarities = (shot_text @ video).tolist()
        return _spearman(text_similarities, video_similarities), truncation

    def _model_text_max_length(self) -> int:
        config = getattr(self._encoder, "config", None)
        text_config = getattr(config, "text_config", None)
        value = getattr(text_config, "max_position_embeddings", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        tokenizer = getattr(self._processor, "tokenizer", None)
        value = getattr(tokenizer, "model_max_length", None)
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 1_000_000:
            return value
        return 77

    def _text_truncation(self, prompts: list[str], max_length: int) -> dict[str, Any]:
        tokenizer = getattr(self._processor, "tokenizer", None)
        lengths: list[Optional[int]] = []
        if tokenizer is not None:
            for text in prompts:
                try:
                    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
                    ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
                    lengths.append(len(ids) if isinstance(ids, list) else None)
                except Exception:
                    lengths.append(None)
        else:
            lengths = [None] * len(prompts)
        return {
            "max_length": max_length,
            "global_story_strategy": "normalized_mean_of_all_per_shot_text_embeddings",
            "shot_prompts_truncated": [
                index + 1
                for index, length in enumerate(lengths)
                if length is not None and length > max_length
            ],
            "all_shots_represented": True,
            "length_detection_available": all(length is not None for length in lengths),
        }

    def _semantic_scores(
        self,
        prompts: Sequence[str],
        frames: Sequence[Any],
        timestamps: Sequence[float],
        rounds: int,
    ) -> Optional[dict[str, float]]:
        frame_parts = self._frame_parts(frames, timestamps)
        caption_text = self._call_judge(
            "caption",
            [
                {
                    "type": "text",
                    "text": (
                        "Describe the ordered sampled frames as chronological observations. "
                        "Return JSON with exactly one key, captions, whose value is a list of "
                        "concise strings in the same order as the labelled timestamps."
                    ),
                },
                *frame_parts,
            ],
        )
        captions = self._parse_captions(caption_text, len(frames))
        if captions is None:
            return None
        reasoning_text = self._call_judge(
            "reasoning",
            [
                {
                    "type": "text",
                    "text": (
                        "Compare the intended ordered shot prompts with the observed chronological "
                        "captions. Analyze narrative flow, cause/effect continuity, and whether "
                        "people, creatures, objects, and places persist consistently. Ignore local "
                        "image polish. Return JSON with exactly one non-empty string key: analysis.\n"
                        f"INTENDED={json.dumps(list(prompts), ensure_ascii=False)}\n"
                        f"OBSERVED={json.dumps(captions, ensure_ascii=False)}"
                    ),
                }
            ],
        )
        analysis = self._parse_analysis(reasoning_text)
        if analysis is None:
            return None
        rows = []
        for round_index in range(rounds):
            score_text = self._call_judge(
                "scoring",
                [
                    {
                        "type": "text",
                        "text": (
                            "Using the ordered frames and prior analysis, score only long-range "
                            "properties from 0 to 100. narrative measures coherent progression; "
                            "causality measures plausible dependencies and consequences; identity "
                            "measures persistent entities across separated moments. Return only a "
                            "JSON object with exactly numeric keys narrative, causality, identity. "
                            f"Independent scoring round {round_index + 1}.\n"
                            f"INTENDED={json.dumps(list(prompts), ensure_ascii=False)}\n"
                            f"CAPTIONS={json.dumps(captions, ensure_ascii=False)}\n"
                            f"ANALYSIS={analysis}"
                        ),
                    },
                    *frame_parts,
                ],
            )
            parsed = self._parse_scores(score_text)
            if parsed is None:
                return None
            rows.append(parsed)
        return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}

    def _call_judge(self, stage: str, content: list[dict[str, Any]]) -> Any:
        response = self._client.chat.completions.create(
            model=str(self.config["judge_model"]),
            messages=[
                {"role": "system", "content": f"Long-context video evaluation stage: {stage}."},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=int(self.config.get("max_tokens", 1000)),
        )
        return response.choices[0].message.content

    @staticmethod
    def _frame_parts(frames: Sequence[Any], timestamps: Sequence[float]) -> list[dict[str, Any]]:
        import cv2

        parts: list[dict[str, Any]] = []
        for index, (frame, timestamp) in enumerate(zip(frames, timestamps), start=1):
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not ok:
                raise ValueError("frame JPEG encoding failed")
            data = base64.b64encode(encoded.tobytes()).decode("ascii")
            parts.extend(
                [
                    {"type": "text", "text": f"FRAME {index} @ {timestamp:.3f}s"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}},
                ]
            )
        return parts

    @staticmethod
    def _parse_captions(text: Any, expected: int) -> Optional[list[str]]:
        payload = LongCodeModule._json_object(text)
        if payload is None or set(payload) != {"captions"}:
            return None
        captions = payload["captions"]
        if not isinstance(captions, list) or len(captions) != expected:
            return None
        if any(not isinstance(value, str) or not value.strip() for value in captions):
            return None
        return [value.strip() for value in captions]

    @staticmethod
    def _parse_analysis(text: Any) -> Optional[str]:
        payload = LongCodeModule._json_object(text)
        if payload is None or set(payload) != {"analysis"}:
            return None
        value = payload["analysis"]
        return value.strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _parse_scores(text: Any) -> Optional[dict[str, float]]:
        payload = LongCodeModule._json_object(text)
        keys = {"narrative", "causality", "identity"}
        if payload is None or set(payload) != keys:
            return None
        result: dict[str, float] = {}
        for key in keys:
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 100.0:
                return None
            result[key] = number
        return result

    @staticmethod
    def _json_object(text: Any) -> Optional[dict[str, Any]]:
        if not isinstance(text, str):
            return None
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _valid_scores(values: Sequence[float]) -> bool:
        return all(math.isfinite(value) and 0.0 <= value <= 100.0 for value in values)

    @staticmethod
    def _write_metrics(sample: Sample, values: dict[str, float]) -> None:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        for field, value in values.items():
            setattr(sample.quality_metrics, field, value)

    @staticmethod
    def _clear_outputs(sample: Sample) -> None:
        sample.metadata.pop("long_code", None)
        metrics = sample.quality_metrics
        if metrics is None:
            return
        for field in OUTPUT_FIELDS:
            if hasattr(metrics, field):
                setattr(metrics, field, None)
        lineage = getattr(metrics, "metric_provenance", None)
        if isinstance(lineage, dict):
            for field in OUTPUT_FIELDS:
                lineage.pop(field, None)
        backends = getattr(metrics, "metric_backends", None)
        if isinstance(backends, dict):
            backends.pop("long_code", None)

    def on_dispose(self) -> None:
        """Close an owned hosted client and release model references."""
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    logger.warning("Long-CODE client cleanup failed: %s", exc)
        self._client = None
        self._processor = None
        self._encoder = None
        self._encoder_backend = None
        self._semantic_backend = None
        self._backend = None
        super().on_dispose()

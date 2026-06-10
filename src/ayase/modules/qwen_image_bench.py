"""Qwen-Image-Bench judge metrics for text-to-image generations.

Scores image/prompt pairs across Quality, Aesthetics, Alignment,
Real-world Fidelity, and Creative Generation on the benchmark 0-100 scale.
Requires a caption or sidecar prompt text and skips videos.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


SCORE_MAP = {0: 0.0, 1: 60.0, 2: 100.0}

DIMENSION_HIERARCHY: Dict[str, Dict[str, List[str]]] = {
    "Quality": {
        "Realism": ["Physical Logic", "Material Texture"],
        "Detail": ["Noise", "Edge Clarity", "Naturalness"],
        "Resolution": ["Resolution"],
    },
    "Aesthetics": {
        "Composition": ["Composition"],
        "Color Harmony": ["Color Harmony"],
        "Lighting": ["Lighting & Atmosphere"],
        "Anatomical Portraiture": ["Anatomical Fidelity"],
        "Emotional Expression": ["Emotional Expression"],
        "Style Control": ["Style Control"],
    },
    "Alignment": {
        "Attributes": [
            "Quantity",
            "Facial Expression",
            "Material Properties",
            "Color",
            "Shape",
            "Size",
        ],
        "Actions": ["Contact Interaction", "Non-contact Interaction", "Full-body Action"],
        "Layout": ["2D Space", "3D Space"],
        "Relations": ["Composition Relationship", "Difference/Similarity", "Containment"],
        "Scene": ["Real-world Scene", "Virtual Scene"],
    },
    "Real-world Fidelity": {
        "Fairness": ["Social Bias", "Cultural Fairness"],
        "Safety & Compliance": ["Safety & Compliance"],
        "World Knowledge": [
            "Animals",
            "Objects",
            "Information Visualization",
            "Temporal Characteristics",
            "Cultural Elements",
        ],
    },
    "Creative Generation": {
        "Imagination": ["Imagination"],
        "Feature Matching": ["Feature Matching"],
        "Logical Resolution": ["Logical Resolution"],
        "Text Rendering": ["Text Accuracy", "Text Layout", "Font", "Cross-lingual Generation"],
        "Design Applications": [
            "Graphic Design",
            "Product Design",
            "Spatial Design",
            "Fashion Styling",
            "Game Design",
            "Art Design",
        ],
        "Visual Storytelling": [
            "Cinematic Style",
            "Camera / Lens Style",
            "Storyboard Creation",
            "Shot Sizes",
            "Composition",
            "Angles",
            "Comic Creation",
        ],
    },
}

L3_TO_L2 = {
    dim: {facet: level2 for level2, facets in level2_map.items() for facet in facets}
    for dim, level2_map in DIMENSION_HIERARCHY.items()
}

L3_RENAME = {
    "Creative Generation": {
        "Feature Mapping": "Feature Matching",
    },
}

SYSTEM_PROMPT = (
    "You are an expert evaluator for text-to-image generation quality. "
    "Given an image and the text prompt used to generate it, evaluate the image "
    "on the requested checklist and return structured JSON scores."
)

USER_PROMPT_TEMPLATE = """# Text Prompt Used to Generate the Image
{prompt}

# Evaluation Dimension
{dimension}

# Scoring Rules
- 0 (Fail): clear defect or mismatch.
- 1 (Pass): satisfies the criterion at baseline quality.
- 2 (Excel): exceptionally executed with concrete evidence.
- "N/A": the criterion does not apply to this image/prompt.

# Evaluation Checklist
{checklist}

# Output Format
Respond with a valid JSON object only:
{{
  "Level-2 Dimension": {{
    "Level-3 Dimension": {{"score": 0|1|2|"N/A"}}
  }}
}}
"""


def _normalize_dimension_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


DIMENSION_ALIASES = {
    _normalize_dimension_name(name): name for name in DIMENSION_HIERARCHY
}
DIMENSION_ALIASES.update(
    {
        "realworld": "Real-world Fidelity",
        "realworldfidelity": "Real-world Fidelity",
        "creative": "Creative Generation",
        "creativegeneration": "Creative Generation",
    }
)


def _format_checklist(dimension: str) -> str:
    lines = []
    for level2, facets in DIMENSION_HIERARCHY[dimension].items():
        lines.append(f"## {level2}")
        for facet in facets:
            lines.append(f"- {facet}: score this criterion for the image and prompt.")
    return "\n".join(lines)


def _json_object_candidates(text: str) -> Iterable[str]:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : idx + 1]
                    break
        start = text.find("{", start + 1)


def _extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    text = response_text or ""
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :]
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    for candidate in _json_object_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _map_score(raw_score: Any) -> Optional[float]:
    if isinstance(raw_score, str):
        if raw_score.strip().upper() == "N/A":
            return None
        raw_score = raw_score.strip()

    try:
        value = float(raw_score)
    except (TypeError, ValueError):
        return None

    if value.is_integer() and int(value) in SCORE_MAP:
        return SCORE_MAP[int(value)]
    if 0.0 <= value <= 100.0:
        return value
    return None


def _mean_non_none(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def _fix_score_json(score_json: Dict[str, Any], dimension: str) -> Dict[str, Dict[str, Any]]:
    if not score_json:
        return {}

    mapping = L3_TO_L2.get(dimension, {})
    rename = L3_RENAME.get(dimension, {})
    first_val = next(iter(score_json.values()), None)

    if isinstance(first_val, dict) and "score" in first_val:
        fixed: Dict[str, Dict[str, Any]] = {}
        for level3, score_obj in score_json.items():
            level3 = rename.get(level3, level3)
            level2 = mapping.get(level3, level3)
            fixed.setdefault(level2, {})[level3] = score_obj
        return fixed

    fixed = {}
    for level2, level3_dict in score_json.items():
        if not isinstance(level3_dict, dict):
            continue
        for level3, score_obj in level3_dict.items():
            level3 = rename.get(level3, level3)
            correct_level2 = mapping.get(level3, level2)
            fixed.setdefault(correct_level2, {})[level3] = score_obj
    return fixed


def _compute_dimension_score(score_json: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    level2_scores = {}
    level3_scores = {}

    for level2, level3_dict in score_json.items():
        level3_scores[level2] = {}
        mapped_scores = []
        for level3, score_obj in level3_dict.items():
            raw = score_obj.get("score") if isinstance(score_obj, dict) else score_obj
            mapped = _map_score(raw)
            level3_scores[level2][level3] = mapped
            if mapped is not None:
                mapped_scores.append(mapped)
        level2_scores[level2] = _mean_non_none(mapped_scores)

    return {
        "level1_score": _mean_non_none(level2_scores.values()),
        "level2_scores": level2_scores,
        "level3_scores": level3_scores,
    }


def _aggregate_total_score(dim_results: Dict[str, Dict[str, Any]]) -> Optional[float]:
    return _mean_non_none(
        result.get("level1_score")
        for result in dim_results.values()
        if result is not None
    )


class QwenImageBenchModule(PipelineModule):
    name = "qwen_image_bench"
    description = "Qwen-Image-Bench T2I judge scores across five image-generation dimensions"
    default_config = {
        "model_name": "Qwen/Qwen-Image-Bench",
        "backend": "auto",  # auto | transformers | openai
        "endpoint_url": None,  # OpenAI-compatible /v1/chat/completions endpoint
        "api_key": None,
        "dimensions": "all",
        "prompt": None,
        "device": "auto",
        "dtype": "bfloat16",
        "device_map": "auto",
        "max_new_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "repetition_penalty": 1.05,
        "max_image_size": 1024,
        "resize_to_square": True,
        "warning_threshold": None,
        "trust_remote_code": True,
    }
    models = [
        {
            "id": "Qwen/Qwen-Image-Bench",
            "type": "huggingface",
            "task": "Q-Judger text-to-image evaluation model",
            "size": "27B BF16",
            "notes": "Can also be served through vLLM/SGLang with an OpenAI-compatible endpoint.",
        },
    ]
    metric_info = {
        "qwen_image_bench_quality": "Qwen-Image-Bench Quality L1 score (0-100)",
        "qwen_image_bench_aesthetics": "Qwen-Image-Bench Aesthetics L1 score (0-100)",
        "qwen_image_bench_alignment": "Qwen-Image-Bench Alignment L1 score (0-100)",
        "qwen_image_bench_real_world_fidelity": (
            "Qwen-Image-Bench Real-world Fidelity L1 score (0-100)"
        ),
        "qwen_image_bench_creative_generation": (
            "Qwen-Image-Bench Creative Generation L1 score (0-100)"
        ),
        "qwen_image_bench_overall": "Mean Qwen-Image-Bench L1 score (0-100)",
    }
    metric_groups = {
        "qwen_image_bench_quality": "nr_quality",
        "qwen_image_bench_aesthetics": "aesthetic",
        "qwen_image_bench_alignment": "alignment",
        "qwen_image_bench_real_world_fidelity": "scene",
        "qwen_image_bench_creative_generation": "aesthetic",
        "qwen_image_bench_overall": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "Qwen/Qwen-Image-Bench")
        self.backend_pref = str(self.config.get("backend", "auto")).lower()
        self.endpoint_url = self.config.get("endpoint_url")
        self.api_key = self.config.get("api_key")
        self.device_config = self.config.get("device", "auto")
        self.dtype_config = self.config.get("dtype", "bfloat16")
        self.device_map = self.config.get("device_map", "auto")
        self.max_new_tokens = int(self.config.get("max_new_tokens", 4096))
        self.warning_threshold = self.config.get("warning_threshold")
        self.max_image_size = int(self.config.get("max_image_size", 1024))
        self.resize_to_square = bool(self.config.get("resize_to_square", True))

        self._backend = None
        self._ml_available = False
        self._processor = None
        self._model = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.endpoint_url and self.backend_pref in ("auto", "openai", "endpoint"):
            self._backend = "openai"
            self._ml_available = True
            return

        if self.backend_pref in ("openai", "endpoint"):
            logger.warning("Qwen-Image-Bench endpoint backend requires endpoint_url.")
            return

        try:
            import torch
            from transformers import AutoProcessor

            try:
                from transformers import AutoModelForImageTextToText as ModelClass
            except ImportError:
                from transformers import AutoModelForVision2Seq as ModelClass

            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                resolve_torch_dtype,
            )

            self._device = resolve_torch_device(self.device_config)
            models_dir = self.config.get("models_dir", "models")
            model_id = resolve_model_path(self.model_name, models_dir)
            trust_remote_code = bool(self.config.get("trust_remote_code", True))

            dtype = resolve_torch_dtype(self._device, self.dtype_config)
            model_kwargs = {
                "cache_dir": models_dir,
                "trust_remote_code": trust_remote_code,
                "low_cpu_mem_usage": True,
            }
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            if self.device_map:
                model_kwargs["device_map"] = self.device_map

            self._processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=models_dir,
                trust_remote_code=trust_remote_code,
            )
            self._model = from_pretrained_with_attention(
                ModelClass,
                model_id,
                self.config,
                device=self._device,
                **model_kwargs,
            )
            if not self.device_map and hasattr(self._model, "to"):
                self._model = self._model.to(self._device)
            self._model.eval()
            self._backend = "transformers"
            self._ml_available = True
            logger.info("Qwen-Image-Bench initialized on %s", self._device)
        except ImportError:
            logger.warning(
                "Qwen-Image-Bench unavailable: install/update transformers, torch, "
                "accelerate, and qwen-vl-utils."
            )
        except Exception as e:
            logger.warning("Failed to setup Qwen-Image-Bench: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.is_video:
            return sample
        if not self._ml_available:
            return sample

        prompt = self._get_prompt(sample)
        if not prompt:
            return sample

        try:
            image = self._load_image(sample.path)
            if image is None:
                return sample

            dim_results, raw_outputs = self._score_image(image, sample.path, prompt)
            if not dim_results:
                return sample

            self._store_scores(sample, dim_results)
            self._store_details(sample, dim_results, raw_outputs)

            overall = _aggregate_total_score(dim_results)
            if (
                overall is not None
                and self.warning_threshold is not None
                and overall < self.warning_threshold
            ):
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low Qwen-Image-Bench score: {overall:.2f}/100",
                        details={"qwen_image_bench_overall": overall},
                    )
                )
        except Exception as e:
            logger.warning("Qwen-Image-Bench failed for %s: %s", sample.path, e)

        return sample

    def _get_prompt(self, sample: Sample) -> Optional[str]:
        prompt_override = self.config.get("prompt")
        if prompt_override:
            return str(prompt_override).strip()

        if sample.caption and sample.caption.text.strip():
            return sample.caption.text.strip()

        txt_path = sample.path.with_suffix(".txt")
        if not txt_path.exists():
            return None
        try:
            text = txt_path.read_text(encoding="utf-8").strip()
            return text or None
        except Exception:
            logger.debug("Failed to read Qwen-Image-Bench prompt sidecar: %s", txt_path)
            return None

    def _dimensions(self) -> List[str]:
        configured = self.config.get("dimensions", "all")
        if configured is None or configured == "all":
            return list(DIMENSION_HIERARCHY)
        if isinstance(configured, str):
            configured = [part.strip() for part in configured.split(",")]

        dimensions = []
        for item in configured:
            canonical = DIMENSION_ALIASES.get(_normalize_dimension_name(str(item)))
            if canonical and canonical not in dimensions:
                dimensions.append(canonical)
        return dimensions or list(DIMENSION_HIERARCHY)

    def _load_image(self, path: Path) -> Optional[Image.Image]:
        try:
            image = Image.open(path).convert("RGB")
            if self.max_image_size > 0 and max(image.size) > self.max_image_size:
                if self.resize_to_square:
                    image = image.resize(
                        (self.max_image_size, self.max_image_size),
                        Image.LANCZOS,
                    )
                else:
                    scale = self.max_image_size / max(image.size)
                    size = (
                        max(1, int(round(image.width * scale))),
                        max(1, int(round(image.height * scale))),
                    )
                    image = image.resize(size, Image.LANCZOS)
            image.load()
            return image
        except Exception as e:
            logger.debug("Failed to load Qwen-Image-Bench image %s: %s", path, e)
            return None

    def _score_image(
        self,
        image: Image.Image,
        image_path: Path,
        prompt: str,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        dim_results = {}
        raw_outputs = {}

        for dimension in self._dimensions():
            user_text = USER_PROMPT_TEMPLATE.format(
                prompt=prompt,
                dimension=dimension,
                checklist=_format_checklist(dimension),
            )
            if self._backend == "openai":
                output_text = self._generate_openai(image, user_text)
            else:
                output_text = self._generate_transformers(image, image_path, user_text)
            raw_outputs[dimension] = output_text

            score_json = _extract_json_from_response(output_text)
            if score_json is None:
                logger.debug("Qwen-Image-Bench could not parse %s output", dimension)
                continue

            fixed_json = _fix_score_json(score_json, dimension)
            result = _compute_dimension_score(fixed_json)
            result["raw_score_json"] = fixed_json
            dim_results[dimension] = result

        return dim_results, raw_outputs

    def _generate_transformers(
        self,
        image: Image.Image,
        image_path: Path,
        user_text: str,
    ) -> str:
        import torch

        from ayase.runtime import torch_inference_context

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        inputs = self._prepare_transformers_inputs(messages, image, user_text)
        target_device = self._target_device()
        if hasattr(inputs, "to"):
            inputs = inputs.to(target_device)
        else:
            inputs = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        do_sample = bool(self.config.get("do_sample", False))
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": float(self.config.get("repetition_penalty", 1.05)),
        }
        if do_sample:
            generation_kwargs["temperature"] = float(self.config.get("temperature", 1.0))
            generation_kwargs["top_p"] = float(self.config.get("top_p", 1.0))
            top_k = self.config.get("top_k")
            if top_k is not None:
                generation_kwargs["top_k"] = int(top_k)

        with torch_inference_context(
            str(target_device),
            self.dtype_config,
            bool(self.config.get("amp_enabled", True)),
        ):
            output_ids = self._model.generate(**inputs, **generation_kwargs)

        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        generated_ids = output_ids[:, input_len:] if input_len else output_ids
        if hasattr(self._processor, "batch_decode"):
            return self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def _prepare_transformers_inputs(
        self,
        messages: List[Dict[str, Any]],
        image: Image.Image,
        user_text: str,
    ) -> Any:
        try:
            return self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            pass

        try:
            from qwen_vl_utils import process_vision_info

            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            return self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception:
            prompt_text = f"{SYSTEM_PROMPT}\n\n{user_text}"
            return self._processor(
                text=prompt_text,
                images=image,
                return_tensors="pt",
            )

    def _target_device(self) -> Any:
        if self._model is not None and hasattr(self._model, "device"):
            return self._model.device
        try:
            return next(self._model.parameters()).device
        except Exception:
            return self._device

    def _generate_openai(self, image: Image.Image, user_text: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_to_data_url(image)},
                        },
                    ],
                },
            ],
            "temperature": float(self.config.get("temperature", 0.0)),
            "top_p": float(self.config.get("top_p", 1.0)),
            "max_tokens": self.max_new_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._chat_completions_url(),
            data=data,
            headers=self._openai_headers(),
            method="POST",
        )
        timeout = float(self.config.get("timeout", 600))
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    def _chat_completions_url(self) -> str:
        endpoint = str(self.endpoint_url).rstrip("/")
        if endpoint.endswith("/chat/completions"):
            return endpoint
        if endpoint.endswith("/v1"):
            return f"{endpoint}/chat/completions"
        return f"{endpoint}/v1/chat/completions"

    def _openai_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _image_to_data_url(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _store_scores(self, sample: Sample, dim_results: Dict[str, Dict[str, Any]]) -> None:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        qm = sample.quality_metrics
        qm.qwen_image_bench_quality = self._level1(dim_results, "Quality")
        qm.qwen_image_bench_aesthetics = self._level1(dim_results, "Aesthetics")
        qm.qwen_image_bench_alignment = self._level1(dim_results, "Alignment")
        qm.qwen_image_bench_real_world_fidelity = self._level1(
            dim_results,
            "Real-world Fidelity",
        )
        qm.qwen_image_bench_creative_generation = self._level1(
            dim_results,
            "Creative Generation",
        )
        qm.qwen_image_bench_overall = _aggregate_total_score(dim_results)

    def _store_details(
        self,
        sample: Sample,
        dim_results: Dict[str, Dict[str, Any]],
        raw_outputs: Dict[str, str],
    ) -> None:
        sample.detections.append(
            {
                "type": "qwen_image_bench",
                "backend": self._backend,
                "level1": {
                    dim: result.get("level1_score")
                    for dim, result in dim_results.items()
                },
                "level2": {
                    dim: result.get("level2_scores", {})
                    for dim, result in dim_results.items()
                },
                "raw_score_json": {
                    dim: result.get("raw_score_json", {})
                    for dim, result in dim_results.items()
                },
                "raw_outputs": raw_outputs if self.config.get("store_raw_outputs", False) else None,
            }
        )

    @staticmethod
    def _level1(
        dim_results: Dict[str, Dict[str, Any]],
        dimension: str,
    ) -> Optional[float]:
        result = dim_results.get(dimension)
        if not result:
            return None
        return result.get("level1_score")

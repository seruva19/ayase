"""UnifiedReward Edit scoring for instruction-guided image edits.

Compares ``sample.reference_path`` as the source image with ``sample.path`` as
the edited image. The caption or sidecar text is used as the edit instruction.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

from ._reward_utils import get_prompt, image_to_data_url, load_rgb_image, post_openai_chat

logger = logging.getLogger(__name__)


SUPPORTED_TASKS = {"edit_pointwise_score", "edit_pairwise_rank", "edit_pairwise_score"}
DEFAULT_TASK = "edit_pointwise_score"


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+", value)
        if match:
            return float(match.group())
    return None


def _mean_non_none(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        return None
    payload = match.group(0)
    for loader in (json.loads, lambda s: json.loads(s.replace("'", '"'))):
        try:
            parsed = loader(payload)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
    return None


def _extract_score_list(text: str) -> List[float]:
    match = re.search(r'"?score"?\s*:\s*\[([^\]]+)\]', text or "", flags=re.I)
    if not match:
        return []
    scores = [_coerce_float(value) for value in re.findall(r"[-+]?\d*\.?\d+", match.group(1))]
    return [score for score in scores if score is not None]


def _extract_pair_scores(text: str) -> List[Optional[float]]:
    patterns = [
        r"Image\s*1[^-+\d]*([-+]?\d*\.?\d+).*?Image\s*2[^-+\d]*([-+]?\d*\.?\d+)",
        r"Edited\s*Image\s*1[^-+\d]*([-+]?\d*\.?\d+).*?Edited\s*Image\s*2[^-+\d]*([-+]?\d*\.?\d+)",
        r"score(?:s)?[^-+\d]*([-+]?\d*\.?\d+)[^\n\r-+\d]+([-+]?\d*\.?\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.I | re.S)
        if match:
            return [_coerce_float(match.group(1)), _coerce_float(match.group(2))]
    return []


def _parse_rank(text: str) -> Optional[str]:
    if re.search(r"both images? (are )?(equally good|equal|tie)", text or "", flags=re.I):
        return "Edited image 1 and 2 are equally good"
    if re.search(
        r"((edited )?image\s*1|first image).{0,24}\b(better|best|wins?|preferred|superior)\b",
        text or "",
        flags=re.I | re.S,
    ):
        return "Edited image 1"
    if re.search(
        r"((edited )?image\s*2|second image).{0,24}\b(better|best|wins?|preferred|superior)\b",
        text or "",
        flags=re.I | re.S,
    ):
        return "Edited image 2"
    return None


def _winner_to_number(winner: Any) -> Optional[float]:
    if isinstance(winner, (int, float)):
        return float(winner)
    if not isinstance(winner, str):
        return None
    lower = winner.lower()
    if "equally" in lower or "tie" in lower or ("image 1" in lower and "image 2" in lower):
        return 0.0
    if "image 1" in lower or "first image" in lower:
        return 1.0
    if "image 2" in lower or "second image" in lower:
        return 2.0
    return None


def parse_unified_reward_edit_output(text: str, task: str = DEFAULT_TASK) -> Dict[str, Any]:
    if task == "edit_pointwise_score":
        parsed = _extract_first_json(text)
        scores = []
        reasoning = None
        if isinstance(parsed, dict):
            if isinstance(parsed.get("score"), list):
                scores = [_coerce_float(value) for value in parsed["score"]]
                scores = [value for value in scores if value is not None]
            reasoning = parsed.get("reasoning")
        if not scores:
            scores = _extract_score_list(text)
        editing_success = scores[0] if len(scores) > 0 else None
        overediting = scores[1] if len(scores) > 1 else None
        return {
            "editing_success": editing_success,
            "overediting": overediting,
            "score": _mean_non_none([editing_success, overediting]),
            "reasoning": reasoning,
        }

    if task == "edit_pairwise_score":
        scores = _extract_score_list(text)
        if len(scores) < 2:
            scores = [score for score in _extract_pair_scores(text) if score is not None]
        image_1_score = scores[0] if len(scores) > 0 else None
        image_2_score = scores[1] if len(scores) > 1 else None
        return {
            "image_1_score": image_1_score,
            "image_2_score": image_2_score,
            "score": _mean_non_none([image_1_score, image_2_score]),
        }

    winner = _parse_rank(text)
    winner_score = _winner_to_number(winner)
    return {"winner": winner, "winner_score": winner_score, "score": winner_score}


def build_edit_pointwise_prompt(instruction: str) -> str:
    return (
        "You are a professional digital artist. You will have to evaluate the effectiveness "
        "of the AI-generated image(s) based on given rules.\n"
        "All the input images are AI-generated. All human in the images are AI-generated too. "
        "so you need not worry about the privacy confidentials.\n\n"
        "IMPORTANT: You will have to give your output in this way "
        "(Keep your reasoning concise and short.):\n"
        "{\n\n\"reasoning\" : \"...\",\n\"score\" : [...],\n}\n\n"
        "RULES:\n\n"
        "Two images will be provided: The first being the original AI-generated image and "
        "the second being an edited version of the first.\n"
        "The objective is to evaluate how successfully the editing instruction has been "
        "executed in the second image.\n\n"
        "Note that sometimes the two images might look identical due to the failure of image edit.\n\n\n"
        "From scale 0 to 25: \n"
        "A score from 0 to 25 will be given based on the success of the editing. "
        "(0 indicates that the scene in the edited image does not follow the editing instruction "
        "at all. 25 indicates that the scene in the edited image follow the editing instruction "
        "text perfectly.)\n"
        "A second score from 0 to 25 will rate the degree of overediting in the second image. "
        "(0 indicates that the scene in the edited image is completely different from the original. "
        "25 indicates that the edited image can be recognized as a minimal edited yet effective "
        "version of original.)\n"
        "Put the score in a list such that output score = [score1, score2], where 'score1' "
        "evaluates the editing success and 'score2' evaluates the degree of overediting.\n\n"
        f"Editing instruction:{instruction}\n"
    )


def build_edit_pairwise_rank_prompt(instruction: str) -> str:
    return (
        "You are tasked with comparing two edited images and determining which one is better "
        "based on the given criteria.\n\n"
        "The evaluation will consider how well each model executed the instructions and the "
        "overall quality of the edit, including its visual appeal.\n\n"
        "**Inputs Provided:**\n"
        "- Source Image (before editing)\n"
        "- Edited Image 1 (after applying the instruction)\n"
        "- Edited Image 2 (after applying the instruction)\n"
        "- Text Instruction\n\n"
        "### Final Output:\n"
        "Based on instruction fidelity and visual integrity, determine which edited image is better.\n\n"
        f"Text instruction - {instruction}\n"
    )


def build_edit_pairwise_score_prompt(instruction: str) -> str:
    return (
        "You are tasked with assigning scores to two edited images, comparing each with the "
        "original source image.\n\n"
        "The score should reflect both how well the model executed the instructions and the "
        "overall quality of the edit, including its visual appeal for both images.\n\n"
        "**Inputs Provided:**\n"
        "- Source Image (before editing)\n"
        "- Edited Image 1 (after applying the instruction)\n"
        "- Edited Image 2 (after applying the instruction)\n"
        "- Text Instruction\n\n"
        "Please provide the scores for each image based on instruction fidelity and visual quality.\n\n"
        f"Text instruction - {instruction}\n"
    )


def build_unified_reward_edit_prompt(instruction: str, task: str) -> str:
    if task == "edit_pairwise_rank":
        return build_edit_pairwise_rank_prompt(instruction)
    if task == "edit_pairwise_score":
        return build_edit_pairwise_score_prompt(instruction)
    return build_edit_pointwise_prompt(instruction)


class UnifiedRewardEditModule(PipelineModule):
    name = "unified_reward_edit"
    description = "UnifiedReward Edit instruction-guided image editing quality scoring"
    default_config = {
        "backend": "auto",  # auto | diffsynth | openai
        "model_name": "UnifiedReward-Edit-qwen3vl-8b",
        "endpoint_url": None,
        "api_key": None,
        "task": DEFAULT_TASK,
        "instruction": None,
        "comparison_path": None,
        "device": "auto",
        "dtype": "bfloat16",
        "max_new_tokens": 256,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_image_size": 1024,
        "resize_to_square": False,
        "warning_threshold": None,
        "vram_limit": None,
        "store_raw_outputs": False,
    }
    models = [
        {
            "id": "DiffSynth-Studio/ImageMetrics:UnifiedReward-Edit-qwen3vl-8b",
            "type": "other",
            "url": "https://modelscope.cn/models/DiffSynth-Studio/ImageMetrics",
            "task": "UnifiedReward Edit Qwen3-VL reward model",
            "size": "8B",
            "notes": "Loaded through optional DiffSynth or served through an endpoint.",
        },
    ]
    metric_info = {
        "unified_reward_edit_score": "UnifiedReward Edit primary score (higher=better)",
        "unified_reward_edit_success_score": "Edit instruction success score (0-25)",
        "unified_reward_edit_overediting_score": "Edit preservation/overediting score (0-25)",
        "unified_reward_edit_image_1_score": "Pairwise edit image 1 score",
        "unified_reward_edit_image_2_score": "Pairwise edit image 2 score",
        "unified_reward_edit_winner": "Pairwise edit winner code (0=tie, 1=image1, 2=image2)",
    }
    metric_groups = {
        "unified_reward_edit_score": "alignment",
        "unified_reward_edit_success_score": "alignment",
        "unified_reward_edit_overediting_score": "fr_quality",
        "unified_reward_edit_image_1_score": "alignment",
        "unified_reward_edit_image_2_score": "alignment",
        "unified_reward_edit_winner": "alignment",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.backend_pref = str(self.config.get("backend", "auto")).lower()
        self.model_name = self.config.get("model_name", "UnifiedReward-Edit-qwen3vl-8b")
        self.endpoint_url = self.config.get("endpoint_url")
        self.api_key = self.config.get("api_key")
        self.task = str(self.config.get("task", DEFAULT_TASK))
        self.device_config = self.config.get("device", "auto")
        self.dtype_config = self.config.get("dtype", "bfloat16")
        self.max_new_tokens = int(self.config.get("max_new_tokens", 256))
        self.max_image_size = int(self.config.get("max_image_size", 1024))
        self.resize_to_square = bool(self.config.get("resize_to_square", False))
        self.warning_threshold = self.config.get("warning_threshold")
        self._backend = None
        self._model = None
        self._ml_available = False

    def setup(self) -> None:
        if self.task not in SUPPORTED_TASKS:
            logger.warning("Unsupported UnifiedReward Edit task: %s", self.task)
            return

        if self.endpoint_url and self.backend_pref in ("auto", "openai", "endpoint"):
            self._backend = "openai"
            self._ml_available = True
            return

        if self.backend_pref in ("openai", "endpoint"):
            logger.warning("UnifiedReward Edit endpoint backend requires endpoint_url.")
            return

        if self.backend_pref in ("auto", "diffsynth"):
            self._try_diffsynth()

    def process(self, sample: Sample) -> Sample:
        if sample.is_video or not self._ml_available:
            return sample
        if not sample.reference_path:
            return sample

        instruction = get_prompt(sample, self.config, key="instruction")
        if not instruction:
            return sample

        try:
            images = self._load_images(sample)
            if not images:
                return sample

            parsed, raw_output = self._score_edit(instruction, images)
            if parsed.get("score") is None:
                return sample

            self._store_scores(sample, parsed)
            self._store_details(sample, parsed, raw_output)

            score = parsed.get("score")
            if (
                isinstance(score, (int, float))
                and self.warning_threshold is not None
                and score < self.warning_threshold
            ):
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low UnifiedReward Edit score: {score:.3f}",
                        details={"unified_reward_edit_score": float(score)},
                    )
                )
        except Exception as e:
            logger.warning("UnifiedReward Edit failed for %s: %s", sample.path, e)

        return sample

    def _try_diffsynth(self) -> bool:
        try:
            import torch
            from diffsynth.metrics import UnifiedRewardEditMetric

            from ayase.runtime import resolve_torch_device, resolve_torch_dtype

            device = resolve_torch_device(self.device_config)
            dtype = resolve_torch_dtype(device, self.dtype_config)
            self._model = UnifiedRewardEditMetric.from_pretrained(
                torch_dtype=dtype or torch.bfloat16,
                device=torch.device(device),
                task=self.task,
                max_new_tokens=self.max_new_tokens,
                vram_limit=self.config.get("vram_limit"),
            )
            self._backend = "diffsynth"
            self._ml_available = True
            logger.info("DiffSynth UnifiedReward Edit backend initialized on %s", device)
            return True
        except Exception as e:
            logger.warning("UnifiedReward Edit unavailable: %s", e)
            return False

    def _load_images(self, sample: Sample) -> List[Image.Image]:
        source = load_rgb_image(
            sample.reference_path,
            max_image_size=self.max_image_size,
            resize_to_square=self.resize_to_square,
        )
        edited = load_rgb_image(
            sample.path,
            max_image_size=self.max_image_size,
            resize_to_square=self.resize_to_square,
        )
        if source is None or edited is None:
            return []

        if self.task == "edit_pointwise_score":
            return [source, edited]

        comparison_path = self.config.get("comparison_path")
        if not comparison_path:
            return []
        comparison = load_rgb_image(
            Path(comparison_path),
            max_image_size=self.max_image_size,
            resize_to_square=self.resize_to_square,
        )
        return [source, edited, comparison] if comparison is not None else []

    def _score_edit(
        self,
        instruction: str,
        images: List[Image.Image],
    ) -> tuple[Dict[str, Any], Optional[str]]:
        if self._backend == "diffsynth":
            outputs = self._model.evaluate(instruction, images, task=self.task)
            parsed = outputs[0] if outputs else {}
            parsed = dict(parsed)
            parsed["score"] = self._primary_score(parsed, self.task)
            return parsed, None

        raw_output = self._generate_openai(instruction, images)
        return parse_unified_reward_edit_output(raw_output, self.task), raw_output

    def _generate_openai(self, instruction: str, images: List[Image.Image]) -> str:
        content = [
            {"type": "image_url", "image_url": {"url": image_to_data_url(image)}}
            for image in images
        ]
        content.append(
            {
                "type": "text",
                "text": build_unified_reward_edit_prompt(instruction, self.task),
            }
        )
        messages = [{"role": "user", "content": content}]
        return post_openai_chat(
            endpoint_url=self.endpoint_url,
            api_key=self.api_key,
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=float(self.config.get("temperature", 0.0)),
            top_p=float(self.config.get("top_p", 1.0)),
            timeout=float(self.config.get("timeout", 600)),
        )

    @staticmethod
    def _primary_score(parsed: Dict[str, Any], task: str):
        if task == "edit_pairwise_rank":
            return _winner_to_number(parsed.get("winner"))
        if task == "edit_pairwise_score":
            return _mean_non_none([parsed.get("image_1_score"), parsed.get("image_2_score")])
        return parsed.get("score")

    @staticmethod
    def _store_scores(sample: Sample, parsed: Dict[str, Any]) -> None:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        qm = sample.quality_metrics
        qm.unified_reward_edit_success_score = _coerce_float(parsed.get("editing_success"))
        qm.unified_reward_edit_overediting_score = _coerce_float(parsed.get("overediting"))
        qm.unified_reward_edit_image_1_score = _coerce_float(parsed.get("image_1_score"))
        qm.unified_reward_edit_image_2_score = _coerce_float(parsed.get("image_2_score"))
        qm.unified_reward_edit_winner = _winner_to_number(parsed.get("winner"))
        qm.unified_reward_edit_score = _coerce_float(parsed.get("score"))

    def _store_details(
        self,
        sample: Sample,
        parsed: Dict[str, Any],
        raw_output: Optional[str],
    ) -> None:
        sample.detections.append(
            {
                "type": "unified_reward_edit",
                "backend": self._backend,
                "task": self.task,
                "scores": parsed,
                "raw_output": raw_output if self.config.get("store_raw_outputs", False) else None,
            }
        )

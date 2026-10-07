"""Evaluate whether object-level changes in an edited image follow an instruction.

For an image-only sample, ``sample.reference_path`` is the original/source
image and ``sample.path`` is its edited result. The requested edit comes from
config ``instruction``, then ``sample.caption.text``, then the edited image's
``.txt`` sidecar. DICE first detects ADD/REMOVE/EDIT changes from the ordered
source/edited pair, then judges each localized change against that instruction.
``dice_edit_coherence_score`` is the coherent-change fraction in [0, 1]
(higher is better); no detected changes score 0, while incomplete model output
leaves the metric unset.

The backend uses the official DICE Idefics3-8B difference-detector and
coherence adapters. Coherence crops each image to its centered square, resizes
to 512x512, and marks the detected region; setup requires the model snapshots
and substantial memory.

Primary source: https://github.com/aimagelab/DICE
"""

from __future__ import annotations

import gc
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageOps

from ayase.config import download_hf_snapshot
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

from ._reward_utils import get_prompt, load_rgb_image

logger = logging.getLogger(__name__)

DIFFERENCE_REPO = "aimagelab/DICE_differencedet_Idefics"
COHERENCE_REPO = "aimagelab/DICE_coherence_Idefics"
BASE_REPO = "HuggingFaceM4/Idefics3-8B-Llama3"

_DIFFERENCE_MODEL_DIR = (
    "model_based_tuned_stage1/image_first_after15k_after_lvis_idefics"
)
_DIFFERENCE_ADAPTER_DIR = "lora_tuned_stage2/checkpoint-15000"
_COHERENCE_ADAPTER_DIR = "lora_tuned/checkpoint-550"

# Verbatim prompts from editing-evaluation/editing_evaluation/prompts/prompts.py
# (aimagelab/DICE, ICCV 2025).
_DIFFERENCE_PROMPT = """
     You are a system that detects differences between two images.
    - You need to extract the elements that are changed in the second image with respect to the first one.   
	- Create a new entry for each distinct change.
	- For each entry, use this format: "<CHANGE_COMMAND>: <CHANGED_ELEMENT>, (<BOUNDING_BOX>)".
	- CHANGE_COMMAND:
                - ADD: If a new element appears in the second image that was not present in the first.
                - REMOVE: If an element from the first image is missing in the second.
                - EDIT: If an element in the second image is different but in the same location of another element of the first image.
	- CHANGED_ELEMENT: Describe the element that has changed.
	- BOUNDING_BOX: Use normalized coordinates [x0, y0, x1, y1] for the changed element's position in the second image, where (x0, y0) is the top-left corner, and (x1, y1) is the bottom-right corner. The coordinates should be scaled between 0 and 1, with 0 representing one edge of the image and 1 the opposite edge."""  # PRETRAIN_PROMPT_NOCHANGE

_COHERENCE_SYSTEM_PROMPT = """
You are evaluating if a specific change detected by an AI vision model matches the request in the original edit prompt.

## Task 
Determine if the detected change, as described and bounded by the provided colored bbox, matches the request in the original edit prompt. 
A match is valid only if the localized detected change is 100% compatible with the requested prompt. 
Any unwanted modification of the original image (even small) should avoid a match.

## Context
- The original image and the edited image are provided, in this order. The edited image is the original with some changes applied. Focus only on the area specified by the bbox in the detected change.
- Another AI model has detected a change in the image, including its bbox.
    - ADD means that an object is only added in the edited image (on the background).
    - EDIT means that an object is substituted with another one in the edited image.
    - REMOVE means that an object is removed in the edited image.
- Be strict: An EDIT means that an object has been removed and subsituted with anotherone, ensure nothing was removed unless explicitly stated in the prompt. If an object has been removed unexpectedly then you should say NO.

## Example Response
- Reasoning: <REASONING>
- Decision: "YES" or "NO"
    """  # PROMPT_DIFFERENCE_COHERENCE_SYSTEM

_COHERENCE_USER_TEMPLATE = """
## Instructions
1. The original edit prompt is: {SUBTSITUTE_PROMPT}
2. The detected change to evaluate is: {SUBTSITUTE_CHANGE}
3. Use only the text and the observations from the specified bbox area (colored) in both the original and edited images to decide if the specific detected change aligns with the original edit prompt.

Images will follow.
"""  # PROMPT_DIFFERENCE_COHERENCE

_CHANGE_RE = re.compile(
    r"\b(ADD|REMOVE|EDIT)\s*:\s*"
    r"(.+?)\s*,\s*(?:BOUNDING[\s_]*BOX\s*:\s*)?"
    r"[\[(]\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*"
    r"(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*[\])]",
    re.IGNORECASE,
)
_ANSWER_RE = re.compile(r"\b(?:answer|decision)\s*:\s*[\"']?(YES|NO)\b", re.IGNORECASE)


class DICEEditModule(PipelineModule):
    """Evaluate whether localized source-to-edited changes follow an instruction."""

    name = "dice_edit"
    provenance = "adapted"
    sources = {
        "dice_edit_coherence_score": "DICE (ICCV 2025), official aimagelab/DICE_*_Idefics weights and prompts — https://github.com/aimagelab/DICE",
    }
    deviations = {
        "dice_edit_coherence_score": "official weights and verbatim prompts; the final fraction of coherent edits (and 0 when no edits) is own aggregation — the official example emits no single number; box markup via PIL instead of upstream's matplotlib render",
    }
    description = "DICE object-level instruction-guided image-edit coherence (ICCV 2025)"
    default_config = {
        "models_dir": "models",
        "device": "auto",
        "dtype": "bfloat16",
        "instruction": None,
        "processor_longest_edge": 1456,
        "max_new_tokens": 500,
        "warning_threshold": None,
        "store_raw_outputs": False,
    }
    required_packages = ["torch", "transformers", "peft", "huggingface_hub", "Pillow"]
    models = [
        {
            "id": DIFFERENCE_REPO,
            "type": "huggingface",
            "task": "DICE object-level difference detector and stage-2 LoRA",
            "size": "~20 GB",
            "vram": "~20 GB in bfloat16",
            "auto_download": True,
        },
        {
            "id": COHERENCE_REPO,
            "type": "huggingface",
            "task": "DICE edit-coherence LoRA",
            "size": "~2.8 GB",
            "auto_download": True,
        },
        {
            "id": BASE_REPO,
            "type": "huggingface",
            "task": "Idefics3-8B base for DICE coherence estimation",
            "size": "~17 GB",
            "vram": "~20 GB in bfloat16",
            "auto_download": True,
        },
    ]
    metric_info = {
        "dice_edit_coherence_score": (
            "Fraction of DICE-detected object changes judged instruction-coherent "
            "(0-1, higher=better)"
        )
    }
    metric_groups = {"dice_edit_coherence_score": "alignment"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._backend: Optional[str] = None
        self._torch: Any = None
        self._processor: Any = None
        self._device: Any = None
        self._dtype: Any = None
        self._difference_root: Optional[Path] = None
        self._coherence_root: Optional[Path] = None
        self._base_root: Optional[Path] = None

    def setup(self) -> None:
        try:
            import torch
            import transformers  # noqa: F401
            import peft  # noqa: F401

            self._torch = torch
            configured_device = str(self.config.get("device", "auto"))
            if configured_device == "auto":
                configured_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = torch.device(configured_device)
            dtype_name = str(self.config.get("dtype", "bfloat16"))
            self._dtype = getattr(torch, dtype_name, torch.bfloat16)

            models_dir = str(Path(self.config.get("models_dir", "models")).resolve())
            self._difference_root = download_hf_snapshot(
                DIFFERENCE_REPO,
                models_dir,
                allow_patterns=[
                    f"{_DIFFERENCE_MODEL_DIR}/*",
                    f"{_DIFFERENCE_ADAPTER_DIR}/adapter_config.json",
                    f"{_DIFFERENCE_ADAPTER_DIR}/adapter_model.safetensors",
                    f"{_DIFFERENCE_ADAPTER_DIR}/generation_config.json",
                ],
            )
            self._coherence_root = download_hf_snapshot(
                COHERENCE_REPO,
                models_dir,
                allow_patterns=[
                    f"{_COHERENCE_ADAPTER_DIR}/adapter_config.json",
                    f"{_COHERENCE_ADAPTER_DIR}/adapter_model.safetensors",
                    f"{_COHERENCE_ADAPTER_DIR}/generation_config.json",
                ],
            )
            self._base_root = download_hf_snapshot(BASE_REPO, models_dir)

            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                str(self._base_root),
                size={"longest_edge": int(self.config.get("processor_longest_edge", 1456))},
                local_files_only=True,
            )
            self._backend = "dice"
        except Exception as exc:
            self._backend = None
            logger.warning("DICE edit backend unavailable: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if sample.is_video or self._backend is None or sample.reference_path is None:
            return sample
        instruction = get_prompt(sample, self.config, key="instruction")
        if not instruction:
            return sample

        source = load_rgb_image(Path(sample.reference_path))
        edited = load_rgb_image(sample.path)
        if source is None or edited is None:
            return sample

        try:
            detector = self._load_model(
                self._difference_root / _DIFFERENCE_MODEL_DIR,
                self._difference_root / _DIFFERENCE_ADAPTER_DIR,
            )
            difference_text = self._generate(
                detector,
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": _DIFFERENCE_PROMPT},
                        ],
                    }
                ],
                [source, edited],
                repetition_penalty=1.5,
            )
            changes = parse_dice_changes(difference_text)
            self._release_model(detector)
            del detector

            decisions: List[bool] = []
            coherence_outputs: List[str] = []
            if changes:
                coherence_model = self._load_model(
                    self._base_root,
                    self._coherence_root / _COHERENCE_ADAPTER_DIR,
                )
                for change in changes:
                    marked_source, marked_edited = render_dice_change(source, edited, change)
                    prompt = _COHERENCE_USER_TEMPLATE.replace(
                        "{SUBTSITUTE_PROMPT}", instruction
                    ).replace(
                        "{SUBTSITUTE_CHANGE}",
                        f"{change['operation']}: {change['subject']}",
                    )
                    output = self._generate(
                        coherence_model,
                        [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": _COHERENCE_SYSTEM_PROMPT}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image"},
                                    {"type": "image"},
                                ],
                            },
                        ],
                        [marked_source, marked_edited],
                        add_generation_prompt=False,  # upstream mode="coherence" behavior
                    )
                    decision = parse_dice_decision(output)
                    if decision is not None:
                        decisions.append(decision)
                    coherence_outputs.append(output)
                self._release_model(coherence_model)
                del coherence_model

            # An instruction with no detected change has zero adherence. If a
            # coherence response is malformed, do not emit a misleading partial score.
            score = 0.0 if not changes else None
            if changes and len(decisions) == len(changes):
                score = float(sum(decisions) / len(decisions))
            if score is None:
                logger.warning("DICE returned an unparsable coherence decision for %s", sample.path)
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.dice_edit_coherence_score = score
            sample.detections.append(
                {
                    "type": "dice_edit",
                    "backend": self._backend,
                    "instruction": instruction,
                    "changes": [
                        {**change, "coherent": decisions[index] if index < len(decisions) else None}
                        for index, change in enumerate(changes)
                    ],
                    "raw_difference_output": (
                        difference_text if self.config.get("store_raw_outputs", False) else None
                    ),
                    "raw_coherence_outputs": (
                        coherence_outputs if self.config.get("store_raw_outputs", False) else None
                    ),
                }
            )
            threshold = self.config.get("warning_threshold")
            if threshold is not None and score < float(threshold):
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low DICE edit coherence: {score:.3f}",
                        details={"dice_edit_coherence_score": score},
                    )
                )
        except Exception as exc:
            logger.warning("DICE edit failed for %s: %s", sample.path, exc)
            self._empty_cuda_cache()
        return sample

    def _load_model(self, model_path: Path, adapter_path: Path) -> Any:
        from peft import PeftModel
        from transformers import AutoModelForVision2Seq

        base = AutoModelForVision2Seq.from_pretrained(
            str(model_path),
            torch_dtype=self._dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(self._device)
        model = PeftModel.from_pretrained(
            base,
            str(adapter_path),
            local_files_only=True,
        ).eval()
        return model

    def _generate(
        self,
        model: Any,
        messages: Sequence[Dict[str, Any]],
        images: Sequence[Image.Image],
        repetition_penalty: Optional[float] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        prompt = self._processor.apply_chat_template(
            list(messages),
            add_generation_prompt=add_generation_prompt,
        )
        inputs = self._processor(text=prompt, images=list(images), return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        kwargs: Dict[str, Any] = {
            "max_new_tokens": int(self.config.get("max_new_tokens", 500)),
            "do_sample": False,
        }
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = repetition_penalty
        with self._torch.inference_mode():
            generated = model.generate(**inputs, **kwargs)
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0]

    def _release_model(self, model: Any) -> None:
        del model
        gc.collect()
        self._empty_cuda_cache()

    def _empty_cuda_cache(self) -> None:
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def on_dispose(self) -> None:
        self._processor = None
        self._backend = None
        gc.collect()
        self._empty_cuda_cache()
        super().on_dispose()


def parse_dice_changes(text: str) -> List[Dict[str, Any]]:
    """Parse normalized object-level DICE changes from generated text."""

    changes: List[Dict[str, Any]] = []
    for match in _CHANGE_RE.finditer(text):
        bbox = [float(match.group(index)) for index in range(3, 7)]
        if not all(0.0 <= coordinate <= 1.0 for coordinate in bbox):
            continue
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            continue
        item = {
            "operation": match.group(1).upper(),
            "subject": match.group(2).strip().strip("\"'"),
            "bbox": bbox,
        }
        if item not in changes:
            changes.append(item)
    return changes


def parse_dice_decision(text: str) -> Optional[bool]:
    """Return the final explicit DICE YES/NO decision."""

    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).upper() == "YES"


def _square_resize(image: Image.Image) -> Image.Image:
    side = min(image.size)
    left = (image.width - side) / 2
    top = (image.height - side) / 2
    cropped = image.crop((left, top, left + side, top + side))
    return cropped.resize((512, 512), Image.Resampling.LANCZOS)


def render_dice_change(
    source: Image.Image,
    edited: Image.Image,
    change: Dict[str, Any],
) -> Tuple[Image.Image, Image.Image]:
    """Render the localized colored box used by the DICE coherence stage."""

    source_square = _square_resize(source)
    edited_square = _square_resize(edited)
    color = {"ADD": "red", "EDIT": "green", "REMOVE": "blue"}[change["operation"]]
    target = edited_square if change["operation"] in {"ADD", "EDIT"} else source_square
    draw = ImageDraw.Draw(target)
    bbox = change["bbox"]
    xyxy = tuple(int(round(coordinate * 511)) for coordinate in bbox)
    draw.rectangle(xyxy, outline=color, width=3)
    # evaluation renders matplotlib's default white page border.
    source_square = ImageOps.expand(source_square, border=32, fill="white")
    edited_square = ImageOps.expand(edited_square, border=32, fill="white")
    return source_square, edited_square

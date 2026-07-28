"""Object-level coherence evaluation for instruction-guided image edits.

Implements the two-stage DICE evaluator from ICCV 2025: an Idefics3-based
difference detector localizes ADD/REMOVE/EDIT operations, then a separately
trained coherence model decides whether every localized change is requested by
the edit instruction. The scalar score is the fraction of detected changes
judged coherent (0-1, higher is better).
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

_DIFFERENCE_PROMPT = """Compare the second image with the first and list every
distinct object-level change. Each entry must have exactly this form:
"<COMMAND>: <ELEMENT>, BOUNDING_BOX: [x0, y0, x1, y1]".
COMMAND is ADD, REMOVE, or EDIT. Coordinates are normalized to [0, 1].
ADD means newly present, REMOVE means missing, and EDIT means replaced or
changed at the same location. Return a JSON list of strings and nothing else."""

_COHERENCE_SYSTEM_PROMPT = """Judge whether one localized image change is fully
consistent with the requested edit. The original and edited images follow in
that order; a colored box marks the detected change. ADD means newly present,
EDIT means replaced or changed, and REMOVE means missing. Be strict: any
unrequested modification is incoherent. End with exactly "Answer: YES" or
"Answer: NO"."""

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
                    prompt = (
                        f"Requested edit: {instruction}\n"
                        f"Detected change: {change['operation']}: {change['subject']}\n"
                        "Does this localized change match the requested edit?"
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
    ) -> str:
        prompt = self._processor.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
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

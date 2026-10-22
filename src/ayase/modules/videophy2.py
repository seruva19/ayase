"""Official VideoPhy-2 AutoEval semantic and physical-commonsense ratings.

Runs the authors' mPLUG-Owl-Video checkpoint with its native 32-frame
preprocessing, prompts, and 1-5 rating scale.  The joint indicator follows the
published threshold exactly: semantic adherence >= 4 and physical commonsense
>= 4. The official inference runtime is bundled; the pinned model checkpoint is
cached under models_dir. Unavailable dependencies or weights produce no scores.
"""

import importlib
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ayase.config import download_hf_snapshot, resolve_model_path
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


UPSTREAM_REVISION = "01d366c979e4a82875852adbf427ccc036e7373a"
MODEL_ID = "videophysics/videophy_2_auto"
MODEL_REVISION = "ca38370e1dcc79c99c7017dd8755d0a087299afc"

PROMPT_SA = (
    "The following is a conversation between a curious human and an AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\n"
    'Human: Does this video match the description: "{caption}"? '
    "Please rate the video on a scale from 1 to 5, where 5 indicates a perfect match "
    "and 1 indicates no relevance.\n"
    "AI: "
)
PROMPT_PHYSICS = (
    "The following is a conversation between a curious human and an AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\n"
    "Human: Does this video adhere to the physical laws? "
    "Rate the video on a scale from 1 to 5, where 5 means full compliance and 1 means "
    "significant violations.\n"
    "AI: "
)

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def parse_native_rating(output: str) -> Optional[int]:
    """Parse an AutoEval SA/PC response without inventing a failure score."""
    if not isinstance(output, str):
        return None
    match = re.search(r"(?<!\d)([1-5])(?!\d)", output)
    if match:
        return int(match.group(1))
    lowered = output.lower()
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            return value
    return None


def joint_pass(sa_rating: Optional[int], pc_rating: Optional[int]) -> Optional[float]:
    """Return the official per-item joint indicator, or missing if either axis is absent."""
    if sa_rating is None or pc_rating is None:
        return None
    return float(sa_rating >= 4 and pc_rating >= 4)


class VideoPhy2AutoEvalModule(PipelineModule):
    """VideoPhy-2's released multitask AutoEval model."""

    name = "videophy2_autoeval"
    vendor_components = ("videophy2",)
    description = "Official VideoPhy-2 AutoEval native SA/PC ratings and joint threshold"
    input_type = "video"
    provenance = "published"
    sources = {
        "*": "Bansal et al., VideoPhy-2, arXiv:2503.06800; https://videophy2.github.io/"
    }
    models = [
        {
            "id": MODEL_ID,
            "type": "huggingface",
            "task": "VideoPhy-2 semantic adherence and physical commonsense AutoEval",
            "auto_download": True,
            "notes": (
                f"Official mPLUG-Owl-Video checkpoint revision {MODEL_REVISION}; "
                f"source revision {UPSTREAM_REVISION}"
            ),
        }
    ]
    metric_info = {
        "videophy2_sa_rating": "Native semantic-adherence rating (1-5, higher=better)",
        "videophy2_pc_rating": "Native physical-commonsense rating (1-5, higher=better)",
        "videophy2_joint_pass": "1 iff SA>=4 and PC>=4; missing when either rating is missing",
    }
    metric_groups = {
        "videophy2_sa_rating": "alignment",
        "videophy2_pc_rating": "motion",
        "videophy2_joint_pass": "motion",
    }
    required_packages = ["torch", "transformers", "decord", "sentencepiece"]
    default_config = {
        "model_name": MODEL_ID,
        "models_dir": "models",
        "official_source_dir": None,
        "num_frames": 32,
        "device": "auto",
        "verify_source_revision": True,
        "auto_download": True,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", MODEL_ID)
        self.models_dir = self.config.get("models_dir", "models")
        self.official_source_dir = self.config.get("official_source_dir")
        self.num_frames = int(self.config.get("num_frames", 32))
        self.device = self.config.get("device", "auto")
        self.verify_source_revision = bool(self.config.get("verify_source_revision", True))
        self.auto_download = bool(self.config.get("auto_download", True))
        self._model = None
        self._processor = None
        self._tokenizer = None
        self.active_backend = "unavailable"
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        self.active_backend = "unavailable"
        self._backend = "unavailable"
        try:
            self._load_official_backend()
            self.active_backend = "official"
            self._backend = "official"
        except Exception as exc:
            logger.warning("VideoPhy-2 AutoEval unavailable: %s", exc)

    def _external_source_root(self) -> Path:
        root = Path(self.official_source_dir).expanduser().resolve()
        candidate = root / "VIDEOPHY2"
        if candidate.is_dir():
            root = candidate
        if not (root / "mplug_owl_video" / "modeling_mplug_owl.py").is_file():
            raise FileNotFoundError(f"official VideoPhy-2 inference source not found under {root}")
        if self.verify_source_revision:
            checkout = root.parent if root.name == "VIDEOPHY2" else root
            revision = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if revision != UPSTREAM_REVISION:
                raise RuntimeError(
                    f"VideoPhy-2 source revision {revision!r} does not match {UPSTREAM_REVISION}"
                )
        return root

    def _official_modules(self):
        if not self.official_source_dir:
            modeling = importlib.import_module(
                "ayase.third_party.videophy2.mplug_owl_video.modeling_mplug_owl"
            )
            processing = importlib.import_module(
                "ayase.third_party.videophy2.mplug_owl_video.processing_mplug_owl"
            )
            return modeling, processing

        source_root = self._external_source_root()
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        modeling = importlib.import_module("mplug_owl_video.modeling_mplug_owl")
        processing = importlib.import_module("mplug_owl_video.processing_mplug_owl")
        for imported in (modeling, processing):
            if source_root not in Path(imported.__file__).resolve().parents:
                raise ImportError(
                    f"conflicting mplug_owl_video import resolved outside {source_root}: "
                    f"{imported.__file__}"
                )
        return modeling, processing

    def _resolve_checkpoint(self) -> str:
        explicit_checkpoint = Path(self.model_name).expanduser()
        checkpoint = (
            str(explicit_checkpoint.resolve())
            if explicit_checkpoint.is_dir()
            else resolve_model_path(self.model_name, self.models_dir)
        )
        if checkpoint != self.model_name:
            return checkpoint
        if not self.auto_download:
            raise FileNotFoundError(
                f"checkpoint {self.model_name!r} is not cached under "
                f"models_dir={self.models_dir!r}"
            )
        if self.model_name != MODEL_ID:
            raise ValueError("automatic download is restricted to the official checkpoint")
        return str(
            download_hf_snapshot(
                MODEL_ID,
                self.models_dir,
                revision=MODEL_REVISION,
            )
        )

    def _load_official_backend(self) -> None:
        if self.num_frames != 32:
            raise ValueError("published VideoPhy-2 AutoEval requires num_frames=32")
        import torch
        from ayase.runtime import resolve_torch_device
        from transformers import LlamaTokenizer

        device = resolve_torch_device(self.device)
        if not str(device).startswith("cuda"):
            raise RuntimeError("official VideoPhy-2 AutoEval requires a CUDA device")
        modeling, processing = self._official_modules()
        checkpoint = self._resolve_checkpoint()

        tokenizer = LlamaTokenizer.from_pretrained(checkpoint, local_files_only=True)
        image_processor = processing.MplugOwlImageProcessor.from_pretrained(
            checkpoint, local_files_only=True
        )
        processor = processing.MplugOwlProcessor(image_processor, tokenizer)
        dtype = torch.bfloat16
        model = modeling.MplugOwlForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            device_map={"": "cpu"},
            local_files_only=True,
        )
        model = model.to(device).to(dtype).eval()
        self._tokenizer = tokenizer
        self._processor = processor
        self._model = model

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or self._backend != "official":
            return sample
        caption = sample.caption.text.strip() if sample.caption and sample.caption.text else ""
        pc = self._safe_infer(sample.path, PROMPT_PHYSICS, "PC")
        sa = (
            self._safe_infer(sample.path, PROMPT_SA.format(caption=caption), "SA")
            if caption
            else None
        )

        if pc is None and sa is None:
            return sample
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if pc is not None:
            sample.quality_metrics.videophy2_pc_rating = float(pc)
        if sa is not None:
            sample.quality_metrics.videophy2_sa_rating = float(sa)
        joint = joint_pass(sa, pc)
        if joint is not None:
            sample.quality_metrics.videophy2_joint_pass = joint
        return sample

    def _safe_infer(self, video_path: Path, prompt: str, task: str) -> Optional[int]:
        try:
            return self._infer(video_path, prompt)
        except Exception as exc:
            logger.warning("VideoPhy-2 AutoEval %s failed for %s: %s", task, video_path, exc)
            return None

    def _infer(self, video_path: Path, prompt: str) -> Optional[int]:
        import torch

        inputs = self._processor(
            text=[prompt],
            videos=[str(video_path)],
            num_frames=self.num_frames,
            return_tensors="pt",
        )
        inputs = {
            key: value.bfloat16() if value.dtype == torch.float else value
            for key, value in inputs.items()
        }
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                do_sample=False,
                top_k=1,
                temperature=0.001,
                max_length=256,
            )
        output = self._tokenizer.decode(generated.tolist()[0], skip_special_tokens=True)
        return parse_native_rating(output)

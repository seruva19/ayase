"""World-model instruction, physics, and commonsense evaluation.

Loads the official human-aligned VILA judge from Hugging Face during setup and
evaluates matching WorldModelBench videos natively. Scores preserve the raw
benchmark ranges: instruction 0-3, physics 0-5, commonsense 0-2, total 0-10.
"""

import json
import importlib.machinery
import logging
import re
import sys
import types
import zipfile
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, List, Optional

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

BENCHMARK_REVISION = "00b7aa17a05f9fd1ab5c8f66bcf476d04c9c33bf"
VILA_REVISION = "0f1426e8da9181e6e6653e10bc15f62d515fa2f6"
S2WRAPPER_REVISION = "9c008a37540e761f53574b488979db6e49a64312"
MIRROR_REVISION = "main"
MIRROR_BASE = (
    "https://huggingface.co/AkaneTendo25/ayase-models/resolve/"
    f"{MIRROR_REVISION}/"
)
BENCHMARK_URL = (
    f"{MIRROR_BASE}worldmodelbench/worldmodelbench.json"
)
VILA_SOURCE_URL = f"{MIRROR_BASE}worldmodelbench/vila-source-{VILA_REVISION}.zip"
S2WRAPPER_SOURCE_URL = (
    f"{MIRROR_BASE}worldmodelbench/s2wrapper-source-{S2WRAPPER_REVISION}.zip"
)

PHYSICAL_FIELDS = (
    "worldmodelbench_newton_adherence",
    "worldmodelbench_mass_solid_adherence",
    "worldmodelbench_fluid_adherence",
    "worldmodelbench_penetration_adherence",
    "worldmodelbench_gravity_adherence",
)
COMMON_SENSE_FIELDS = (
    "worldmodelbench_aesthetics_adherence",
    "worldmodelbench_temporal_adherence",
)
PHYSICAL_QUESTIONS = (
    "objects moving without an external force, violating Newton's laws",
    "irregular deformation that violates mass conservation or solid mechanics",
    "liquid flowing in a physically unnatural way",
    "objects passing through one another non-physically",
    "behavior inconsistent with gravity",
)
COMMON_SENSE_QUESTIONS = (
    "poor aesthetics or visibly low-quality content",
    "temporal inconsistency such as flicker or abrupt changes",
)


class WorldModelBenchModule(PipelineModule):
    """Evaluate a WorldModelBench-compatible video set with its official judge."""

    name = "worldmodelbench"
    description = "Official WorldModelBench instruction, physics, and commonsense scores"
    default_config = {
        "model_name": "Efficient-Large-Model/vila-ewm-qwen2-1.5b",
        "model_revision": None,
        "models_dir": "models",
        "benchmark_url": BENCHMARK_URL,
        "vila_source_url": VILA_SOURCE_URL,
        "s2wrapper_source_url": S2WRAPPER_SOURCE_URL,
        "attention_implementation": "sdpa",
        "device": "auto",
        "cot": False,
    }
    required_packages = ["torch"]
    models = [
        {
            "id": "Efficient-Large-Model/vila-ewm-qwen2-1.5b",
            "type": "huggingface",
            "task": "Human-aligned WorldModelBench video judge",
            "auto_download": True,
        },
        {
            "id": "AkaneTendo25/ayase-models",
            "type": "huggingface",
            "url": BENCHMARK_URL,
            "task": "Mirrored benchmark definition and VILA runtime source",
            "auto_download": True,
            "notes": (
                f"WorldModelBench {BENCHMARK_REVISION}; VILA {VILA_REVISION}; "
                f"S2Wrapper {S2WRAPPER_REVISION}"
            ),
        },
    ]
    metric_info = {
        "worldmodelbench_instruction_score": "Instruction following mean (0-3, higher=better)",
        "worldmodelbench_newton_adherence": "Fraction without a Newton-law violation (0-1)",
        "worldmodelbench_mass_solid_adherence": "Fraction without mass/solid-law violation (0-1)",
        "worldmodelbench_fluid_adherence": "Fraction without fluid-law violation (0-1)",
        "worldmodelbench_penetration_adherence": "Fraction without nonphysical penetration (0-1)",
        "worldmodelbench_gravity_adherence": "Fraction without gravity violation (0-1)",
        "worldmodelbench_aesthetics_adherence": "Fraction without poor-aesthetics finding (0-1)",
        "worldmodelbench_temporal_adherence": "Fraction without temporal inconsistency (0-1)",
        "worldmodelbench_physical_score": "Sum of five physical adherence rates (0-5)",
        "worldmodelbench_common_sense_score": "Sum of two commonsense adherence rates (0-2)",
        "worldmodelbench_total_score": "Official raw total (0-10, higher=better)",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._backend: Optional[str] = None
        self._judge: Any = None
        self._llava: Any = None
        self._benchmark: List[Dict[str, Any]] = []

    @staticmethod
    def _extract_source(archive: Path, destination: Path) -> Path:
        marker = destination / ".complete"
        if marker.is_file():
            roots = [path for path in destination.iterdir() if path.is_dir()]
            if roots:
                return roots[0]
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (root / member.filename).resolve()
                try:
                    target.relative_to(root)
                except ValueError as exc:
                    raise ValueError(f"Unsafe VILA archive member: {member.filename}") from exc
            bundle.extractall(root)
        marker.touch()
        roots = [path for path in destination.iterdir() if path.is_dir()]
        if not roots:
            raise FileNotFoundError("VILA source archive contained no source directory")
        return roots[0]

    @staticmethod
    def _ensure_inference_distributed_compatibility() -> None:
        """Provide VILA's unused DeepSpeed comm import during single-device inference."""
        try:
            import deepspeed.comm  # type: ignore[import-not-found]  # noqa: F401

            return
        except ImportError:
            pass

        import torch.distributed as torch_dist

        comm = types.ModuleType("deepspeed.comm")
        for name in dir(torch_dist):
            if not name.startswith("__"):
                setattr(comm, name, getattr(torch_dist, name))

        def init_distributed(*args: Any, **kwargs: Any) -> None:
            if torch_dist.is_initialized():
                return
            backend = kwargs.pop("dist_backend", None) or kwargs.pop("backend", None)
            kwargs.pop("dist_init_required", None)
            torch_dist.init_process_group(backend=backend, *args, **kwargs)

        comm.init_distributed = init_distributed  # type: ignore[attr-defined]
        deepspeed = types.ModuleType("deepspeed")
        deepspeed.__path__ = []  # type: ignore[attr-defined]
        deepspeed.comm = comm  # type: ignore[attr-defined]
        sys.modules.setdefault("deepspeed", deepspeed)
        sys.modules.setdefault("deepspeed.comm", comm)

    @staticmethod
    def _ensure_unused_flash_attention_compatibility(compat_root: Path) -> None:
        """Allow eager InternViT imports when the selected judge uses SigLIP."""
        try:
            import flash_attn  # type: ignore[import-not-found]  # noqa: F401

            return
        except ImportError:
            pass

        metadata = compat_root / "flash_attn-0.0.0.dist-info" / "METADATA"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        if not metadata.is_file():
            metadata.write_text(
                "Metadata-Version: 2.1\nName: flash-attn\nVersion: 0.0.0\n",
                encoding="utf-8",
            )
        if str(compat_root) not in sys.path:
            sys.path.insert(0, str(compat_root))

        def unavailable(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("FlashAttention is unavailable for the unused InternViT backend")

        package = types.ModuleType("flash_attn")
        package.__path__ = []  # type: ignore[attr-defined]
        package.__spec__ = importlib.machinery.ModuleSpec(
            "flash_attn", loader=None, is_package=True
        )
        interface = types.ModuleType("flash_attn.flash_attn_interface")
        interface.__spec__ = importlib.machinery.ModuleSpec(
            "flash_attn.flash_attn_interface", loader=None
        )
        interface.flash_attn_unpadded_qkvpacked_func = unavailable  # type: ignore[attr-defined]
        interface.flash_attn_varlen_qkvpacked_func = unavailable  # type: ignore[attr-defined]
        padding = types.ModuleType("flash_attn.bert_padding")
        padding.__spec__ = importlib.machinery.ModuleSpec(
            "flash_attn.bert_padding", loader=None
        )
        padding.pad_input = unavailable  # type: ignore[attr-defined]
        padding.unpad_input = unavailable  # type: ignore[attr-defined]
        sys.modules.setdefault("flash_attn", package)
        sys.modules.setdefault("flash_attn.flash_attn_interface", interface)
        sys.modules.setdefault("flash_attn.bert_padding", padding)

    @staticmethod
    def _ensure_unused_ps3_compatibility() -> None:
        """Allow eager PS3 registration when the selected judge uses SigLIP."""
        try:
            import ps3  # type: ignore[import-not-found]  # noqa: F401

            return
        except ImportError:
            pass

        from transformers import PretrainedConfig

        class PS3Config(PretrainedConfig):
            model_type = "ps3"

        class PS3VisionConfig(PretrainedConfig):
            model_type = "ps3_vision_model"

        class UnavailablePS3Model:
            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("PS3 is unavailable for the unused PS3 vision backend")

        module = types.ModuleType("ps3")
        module.PS3Config = PS3Config  # type: ignore[attr-defined]
        module.PS3VisionConfig = PS3VisionConfig  # type: ignore[attr-defined]
        module.PS3ImageProcessor = UnavailablePS3Model  # type: ignore[attr-defined]
        module.PS3VisionModel = UnavailablePS3Model  # type: ignore[attr-defined]
        sys.modules.setdefault("ps3", module)

    @staticmethod
    def _ensure_unused_qwen2_fp8_compatibility() -> None:
        """Restore names imported by VILA's unused experimental FP8 classes."""
        from transformers.models.qwen2 import modeling_qwen2

        attention = modeling_qwen2.Qwen2Attention
        if not hasattr(modeling_qwen2, "Qwen2FlashAttention2"):
            modeling_qwen2.Qwen2FlashAttention2 = attention
        if not hasattr(modeling_qwen2, "Qwen2SdpaAttention"):
            modeling_qwen2.Qwen2SdpaAttention = attention
        if not hasattr(modeling_qwen2.Qwen2Model, "_update_causal_mask"):
            def unavailable(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("Legacy causal-mask hook requested by unused FP8 backend")

            modeling_qwen2.Qwen2Model._update_causal_mask = unavailable

    @staticmethod
    def _configure_vila_attention(implementation: str) -> None:
        """Override VILA's hard-coded SigLIP FlashAttention request."""
        from llava.model.multimodal_encoder.siglip import SiglipVisionModel

        if getattr(SiglipVisionModel, "_ayase_attention_override", None) == implementation:
            return
        original = SiglipVisionModel.from_pretrained

        def from_pretrained(model_name_or_path: str, *args: Any, **kwargs: Any) -> Any:
            kwargs["attn_implementation"] = implementation
            return original(model_name_or_path, *args, **kwargs)

        SiglipVisionModel.from_pretrained = staticmethod(from_pretrained)
        SiglipVisionModel._ayase_attention_override = implementation

    def setup(self) -> None:
        try:
            from ayase.config import download_hf_snapshot, download_model_file

            models_dir = str(self.config.get("models_dir", "models"))
            judge_path = download_hf_snapshot(
                str(self.config.get("model_name", self.default_config["model_name"])),
                models_dir,
                revision=self.config.get("model_revision"),
            )
            benchmark_path = download_model_file(
                "worldmodelbench/worldmodelbench.json",
                str(self.config.get("benchmark_url", BENCHMARK_URL)),
                models_dir,
            )
            vila_archive = download_model_file(
                f"worldmodelbench/vila-source-{VILA_REVISION}.zip",
                str(self.config.get("vila_source_url", VILA_SOURCE_URL)),
                models_dir,
            )
            vila_root = self._extract_source(
                vila_archive,
                Path(models_dir) / "worldmodelbench" / f"vila-source-{VILA_REVISION}",
            )
            s2wrapper_archive = download_model_file(
                f"worldmodelbench/s2wrapper-source-{S2WRAPPER_REVISION}.zip",
                str(self.config.get("s2wrapper_source_url", S2WRAPPER_SOURCE_URL)),
                models_dir,
            )
            s2wrapper_root = self._extract_source(
                s2wrapper_archive,
                Path(models_dir)
                / "worldmodelbench"
                / f"s2wrapper-source-{S2WRAPPER_REVISION}",
            )
            if str(s2wrapper_root) not in sys.path:
                sys.path.insert(0, str(s2wrapper_root))
            if str(vila_root) not in sys.path:
                sys.path.insert(0, str(vila_root))
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            if not isinstance(benchmark, list):
                raise ValueError("WorldModelBench definition must be a JSON list")

            self._ensure_inference_distributed_compatibility()
            self._ensure_unused_flash_attention_compatibility(
                Path(models_dir) / "worldmodelbench" / "compat"
            )
            self._ensure_unused_ps3_compatibility()
            self._ensure_unused_qwen2_fp8_compatibility()
            import llava

            self._configure_vila_attention(
                str(self.config.get("attention_implementation", "sdpa"))
            )
            from ayase.runtime import resolve_torch_device

            device = resolve_torch_device(self.config.get("device", "auto"))
            if device == "cuda":
                device = "cuda:0"
            self._judge = llava.load(str(judge_path), device=device)
            self._llava = llava
            self._benchmark = [item for item in benchmark if isinstance(item, dict)]
            self._backend = "vila"
        except Exception as exc:
            self._backend = "unavailable"
            logger.warning("WorldModelBench unavailable: %s", exc)

    def process(self, sample: Sample) -> Sample:
        return sample

    @staticmethod
    def _means(values: Any, width: int) -> Optional[List[float]]:
        if not isinstance(values, list) or not values or len(values) % width:
            return None
        numeric: List[float] = []
        for value in values:
            if isinstance(value, bool):
                numeric.append(float(value))
            elif isinstance(value, (int, float)):
                numeric.append(float(value))
            else:
                return None
        return [fmean(numeric[index::width]) for index in range(width)]

    @classmethod
    def _parse_metrics(cls, payload: Any) -> Dict[str, float]:
        if not isinstance(payload, dict) or not isinstance(payload.get("accs"), dict):
            return {}
        accs = payload["accs"]
        metrics: Dict[str, float] = {}
        instruction = cls._means(accs.get("instruction"), 1)
        physical = cls._means(accs.get("physical_laws"), len(PHYSICAL_FIELDS))
        common = cls._means(accs.get("common_sense"), len(COMMON_SENSE_FIELDS))
        if instruction:
            metrics["worldmodelbench_instruction_score"] = instruction[0]
        if physical:
            metrics.update(zip(PHYSICAL_FIELDS, physical))
            metrics["worldmodelbench_physical_score"] = sum(physical)
        if common:
            metrics.update(zip(COMMON_SENSE_FIELDS, common))
            metrics["worldmodelbench_common_sense_score"] = sum(common)
        aggregate = (
            "worldmodelbench_instruction_score",
            "worldmodelbench_physical_score",
            "worldmodelbench_common_sense_score",
        )
        if all(field in metrics for field in aggregate):
            metrics["worldmodelbench_total_score"] = sum(metrics[field] for field in aggregate)
        return metrics

    def _ask(self, video: Any, prompt: str) -> str:
        if not self.config.get("cot", False):
            prompt += " Give only the requested final score or yes/no answer."
        return str(self._judge.generate_content([video, prompt]))

    @staticmethod
    def _instruction_score(answer: str) -> float:
        matches = re.findall(r"(?:score\s*:\s*)?([0-3](?:\.0+)?)", answer, re.IGNORECASE)
        return float(matches[-1]) if matches else 0.0

    def _evaluate(self, video_path: Path, instruction: str) -> Dict[str, List[Any]]:
        video = self._llava.Video(str(video_path))
        instruction_prompt = (
            f"Judge whether the video follows this instruction: {instruction!r}. "
            "Score 0 if it does not follow it; 1 if only the object or action is right; "
            "2 if it follows the instruction and trends toward the goal; 3 if it completes "
            "the goal precisely. Conclude with 'Score: N'."
        )
        instruction_score = self._instruction_score(self._ask(video, instruction_prompt))
        physical = []
        for question in PHYSICAL_QUESTIONS:
            answer = self._ask(video, f"Does this video show {question}? Conclude yes or no.")
            physical.append("no" in answer.lower())
        common = []
        for question in COMMON_SENSE_QUESTIONS:
            answer = self._ask(video, f"Does this video exhibit {question}? Conclude yes or no.")
            common.append("no" in answer.lower())
        return {
            "instruction": [instruction_score],
            "physical_laws": physical,
            "common_sense": common,
        }

    def post_process(self, all_samples: List[Sample]) -> None:
        if self.pipeline is None or self._backend != "vila":
            return
        by_stem = {sample.path.stem: sample.path for sample in all_samples if sample.is_video}
        accs: Dict[str, List[Any]] = {
            "instruction": [],
            "physical_laws": [],
            "common_sense": [],
        }
        matched = 0
        for item in self._benchmark:
            stem = Path(str(item.get("first_frame", ""))).stem
            video_path = by_stem.get(stem)
            instruction = item.get("text_instruction")
            if video_path is None or not isinstance(instruction, str):
                continue
            try:
                scores = self._evaluate(video_path, instruction)
            except Exception as exc:
                logger.warning("WorldModelBench failed for %s: %s", video_path, exc)
                continue
            matched += 1
            for category, values in scores.items():
                accs[category].extend(values)
        if not matched:
            logger.warning("No input videos matched WorldModelBench benchmark stems")
            return
        for field, value in self._parse_metrics({"accs": accs}).items():
            self.pipeline.add_dataset_metric(field, value)

    def on_dispose(self) -> None:
        self._judge = None
        self._llava = None
        super().on_dispose()

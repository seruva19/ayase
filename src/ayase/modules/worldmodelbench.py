"""Evaluate the WorldModelBench video set with its released VILA judge prompts.

This dataset-level runner matches input-video stems to entries in the pinned
WorldModelBench definition and uses each entry's text instruction. Instruction
following is 0–3; five binary no-violation rates sum to the 0–5 physical score;
two binary no-finding rates sum to the 0–2 commonsense score; their raw total is
0–10. Higher is better for all adherence, category, instruction, and total
outputs. Results are written to ``DatasetStats``, not per-sample metrics.

Only videos whose stems match the benchmark ``first_frame`` identifiers are
evaluated. The backend decodes each full video through the upstream VILA runtime
and parses constrained score/yes-no responses; these judge outputs are benchmark
predictions, not direct physical measurements. Missing assets, unmatched stems,
model/setup failures, or unparsable evaluations are skipped, so coverage must be
checked before comparing dataset aggregates.
"""

import json
import importlib.machinery
import logging
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
    "https://huggingface.co/AkaneTendo25/ayase-assets/resolve/"
    f"{MIRROR_REVISION}/"
)
BENCHMARK_URL = (
    f"{MIRROR_BASE}worldmodelbench/worldmodelbench.json"
)
# Mirrored subset of the VILA tree at VILA_REVISION: the llava package only,
# without the prebuilt CUDA kernel artifacts and training scripts.
VILA_SOURCE_URL = f"{MIRROR_BASE}worldmodelbench/vila-llava-{VILA_REVISION}.zip"
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
# Verbatim question pool and prompt templates from the upstream evaluation.py
# (WorldModelBench-Team/WorldModelBench). Question strings are substituted into
# the templates lower-cased, exactly as upstream does.
PHYSICAL_QUESTIONS = (
    "Violation of Newton's Law: Objects move without any external force.",
    "Violation of the Law of Conservation of Mass or Solid Constitutive Law: Objects deform irregularly.",
    "Violation of Fluid Constitutive Law: Liquids flow in an unnatural manner.",
    "Violation of Non-physical Penetration: Objects unnaturally pass through each other.",
    "Violation of Gravity: Objects behave inconsistently with gravity.",
)
COMMON_SENSE_QUESTIONS = (
    "Poor Aesthetics: Visually unappealing or low-quality content.",
    "Temporal Inconsistency: Noticeable flickering or abrupt changes.",
)
INSTRUCTION_TEMPLATE = """
    Evaluate if this video follows the instruction: '{instruction}'.
    Use the following scoring criteria:

    - 0: The video does not follow the instruction at all.
    - 1: The video includes the correct object but performs the wrong action, or vice versa.
    - 2: The video follows the instruction and shows a tendency toward the intended goal.
    - 3: The video follows the instruction precisely and successfully achieves the goal.

    Let's analyze step-by-step and conclude with 'Score: [score]'.
""".strip()
PHYSICAL_LAWS_TEMPLATE = """
    Watch the video and determine if it shows any '{physical_laws}'
    Let's think step-by-step and conclude with "Yes" or "No".
""".strip()
COMMON_SENSE_TEMPLATE = """
    Does the video exhibit '{common_sense}'?
    Let's think step-by-step and conclude with "Yes" or "No".
""".strip()


class WorldModelBenchModule(PipelineModule):
    """Evaluate a WorldModelBench-compatible video set with its upstream judge."""

    name = "worldmodelbench"
    provenance = "published"
    sources = {
        "worldmodelbench_aesthetics_adherence": "WorldModelBench (VILA judge) — https://huggingface.co/Efficient-Large-Model/vila-ewm-qwen2-1.5b",
        "worldmodelbench_common_sense_score": "WorldModelBench (VILA judge) — https://huggingface.co/Efficient-Large-Model/vila-ewm-qwen2-1.5b",
        "worldmodelbench_fluid_adherence": "WorldModelBench (VILA judge) — https://huggingface.co/Efficient-Large-Model/vila-ewm-qwen2-1.5b",
        "worldmodelbench_gravity_adherence": "WorldModelBench (VILA judge) — https://huggingface.co/Efficient-Large-Model/vila-ewm-qwen2-1.5b",
        "worldmodelbench_instruction_score": "WorldModelBench (VILA judge) — https://huggingface.co/Efficient-Large-Model/vila-ewm-qwen2-1.5b",
        "worldmodelbench_mass_solid_adherence": "WorldModelBench (VILA judge) — https://huggingface.co/Efficient-Large-Model/vila-ewm-qwen2-1.5b",
        "worldmodelbench_newton_adherence": "WorldModelBench (VILA judge) — https://huggingface.co/Efficient-Large-Model/vila-ewm-qwen2-1.5b",
        "worldmodelbench_penetration_adherence": "WorldModelBench (VILA judge) — https://huggingface.co/Efficient-Large-Model/vila-ewm-qwen2-1.5b",
        "worldmodelbench_physical_score": "WorldModelBench (VILA judge) — https://huggingface.co/Efficient-Large-Model/vila-ewm-qwen2-1.5b",
        "worldmodelbench_temporal_adherence": "WorldModelBench (VILA judge) — https://huggingface.co/Efficient-Large-Model/vila-ewm-qwen2-1.5b",
        "worldmodelbench_total_score": "WorldModelBench (VILA judge) — https://huggingface.co/Efficient-Large-Model/vila-ewm-qwen2-1.5b",
    }
    description = "WorldModelBench instruction, physics, and commonsense scores"
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
            "id": "AkaneTendo25/ayase-assets",
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
        "worldmodelbench_total_score": "Raw total (0-10, higher=better)",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._backend: Optional[str] = None
        self._judge: Any = None
        self._llava: Any = None
        self._benchmark: List[Dict[str, Any]] = []

    @staticmethod
    def _vendored(name: str, marker: str) -> Path:
        """Path of an in-tree runtime the benchmark imports.

        Args:
            name (str): Directory under ``ayase.vendor``.
            marker (str): Package that must exist inside it.

        Returns:
            Path: Root to place on ``sys.path``.

        Raises:
            RuntimeError: The vendored tree is missing.
        """
        from ayase import vendor

        root = Path(vendor.__file__).resolve().parent / name
        if not (root / marker).is_dir():
            raise RuntimeError(f"{name} runtime is missing from ayase.vendor")
        return root

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
        """Restore names imported by VILA's unused external-backend FP8 classes."""
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
            from ayase.config import (
                download_hf_snapshot,
                download_model_file,
                resolve_assets_url,
            )

            models_dir = str(self.config.get("models_dir", "models"))
            judge_path = download_hf_snapshot(
                str(self.config.get("model_name", self.default_config["model_name"])),
                models_dir,
                revision=self.config.get("model_revision"),
            )
            benchmark_path = download_model_file(
                "worldmodelbench/worldmodelbench.json",
                resolve_assets_url(
                    str(self.config.get("benchmark_url", BENCHMARK_URL)), self.config
                ),
                models_dir,
            )
            vila_root = self._vendored("vila", "llava")
            s2wrapper_root = self._vendored("s2wrapper", "s2wrapper")
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
        # Upstream evaluate_video(): without --cot, the chain-of-thought lead-in
        # is replaced with "Answer with ..." rather than asking for it.
        if not self.config.get("cot", False):
            prompt = prompt.replace(
                "Let's think step-by-step and conclude with", "Answer with"
            ).replace(
                "Let's analyze step-by-step and conclude with", "Answer with"
            )
        return str(self._judge.generate_content([video, prompt]))

    @staticmethod
    def _instruction_score(answer: str) -> float:
        # Upstream: float(pred.split(":")[-1].strip(" .")) with 0 on failure.
        try:
            score = float(answer.split(":")[-1].strip(" ."))
        except ValueError:
            logger.warning("Could not parse score from prediction: %s", answer)
            score = 0.0
        return score

    def _evaluate(self, video_path: Path, instruction: str) -> Dict[str, List[Any]]:
        video = self._llava.Video(str(video_path))
        instruction_score = self._instruction_score(
            self._ask(video, INSTRUCTION_TEMPLATE.format(instruction=instruction))
        )
        physical = []
        for question in PHYSICAL_QUESTIONS:
            answer = self._ask(
                video, PHYSICAL_LAWS_TEMPLATE.format(physical_laws=question.lower())
            )
            physical.append("no" in answer.lower())
        common = []
        for question in COMMON_SENSE_QUESTIONS:
            answer = self._ask(
                video, COMMON_SENSE_TEMPLATE.format(common_sense=question.lower())
            )
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

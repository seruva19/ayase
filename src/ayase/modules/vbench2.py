"""VBench 2.0 intrinsic-faithfulness evaluation.

Runs the Apache-2.0 VBench 2.0 implementation as an optional dataset-level
backend. No proxy scores are emitted when the upstream package or checkpoints
are unavailable. Higher scores indicate better intrinsic faithfulness.

The upstream tree is vendored in-tree (``ayase.vendor.vbench``) at the pinned
commit: Ayase downloads weights, never code. Only what scoring reaches is kept.
The snapshot is 261 MB, but 142 MB of that is four copies of one LVIS annotation
file inside vendored detector repositories, and the rest is assets, notebooks and
benchmark prompt suites; the evaluator itself needs 14 MB of Python and the
0.4 MB ``VBench2_full_info.json`` it reads. Checkpoints are still fetched.
"""

import json
import logging
import os
import shutil
import sys
import types
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type
from urllib.parse import quote

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

MIRROR_REPO = "AkaneTendo25/ayase-runtime-assets"
MIRROR_REVISION = "main"
MIRROR_RESOLVE_BASE = "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve"
VBENCH_REVISION = "45e79ec14e69a2187202c675d2dbce1a71843d53"
COTRACKER_REVISION = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
#: Repositories the operator checks out; the mirror above holds only weights.
VBENCH_REPOSITORY = "https://github.com/Vchitect/VBench"
COTRACKER_REPOSITORY = "https://github.com/facebookresearch/co-tracker"


DIMENSION_FIELDS = {
    "Human_Anatomy": "vbench2_human_anatomy",
    "Human_Identity": "vbench2_human_identity",
    "Human_Clothes": "vbench2_human_clothes",
    "Diversity": "vbench2_diversity",
    "Composition": "vbench2_composition",
    "Dynamic_Spatial_Relationship": "vbench2_dynamic_spatial_relationship",
    "Dynamic_Attribute": "vbench2_dynamic_attribute",
    "Motion_Order_Understanding": "vbench2_motion_order_understanding",
    "Human_Interaction": "vbench2_human_interaction",
    "Complex_Landscape": "vbench2_complex_landscape",
    "Complex_Plot": "vbench2_complex_plot",
    "Camera_Motion": "vbench2_camera_motion",
    "Motion_Rationality": "vbench2_motion_rationality",
    "Instance_Preservation": "vbench2_instance_preservation",
    "Mechanics": "vbench2_mechanics",
    "Thermotics": "vbench2_thermotics",
    "Material": "vbench2_material",
    "Multi-View_Consistency": "vbench2_multiview_consistency",
}

AGGREGATE_DIMENSIONS = {
    "vbench2_creativity_score": ("Diversity", "Composition"),
    "vbench2_commonsense_score": ("Motion_Rationality", "Instance_Preservation"),
    "vbench2_controllability_score": (
        "Dynamic_Spatial_Relationship",
        "Dynamic_Attribute",
        "Motion_Order_Understanding",
        "Human_Interaction",
        "Complex_Landscape",
        "Complex_Plot",
        "Camera_Motion",
    ),
    "vbench2_human_fidelity_score": (
        "Human_Anatomy",
        "Human_Identity",
        "Human_Clothes",
    ),
    "vbench2_physics_score": (
        "Mechanics",
        "Thermotics",
        "Material",
        "Multi-View_Consistency",
    ),
}

CUSTOM_INPUT_DIMENSIONS = {
    "Human_Anatomy",
    "Human_Identity",
    "Human_Clothes",
    "Diversity",
    "Multi-View_Consistency",
}


class VBench2Module(PipelineModule):
    """Run the VBench 2.0 evaluator after sample processing."""

    name = "vbench2"
    description = "VBench 2.0 18-dimension intrinsic-faithfulness suite"
    default_config = {
        "dimensions": list(DIMENSION_FIELDS),
        "mode": "vbench_standard",
        "videos_path": None,
        "full_info_path": None,
        "output_dir": "reports/vbench2",
        "result_name": "ayase_vbench2",
        "models_dir": "models",
        "model_repo": "Vchitect/VBench-2.0_models",
        "model_revision": None,
        "mirror_revision": MIRROR_REVISION,
        "read_frame": False,
    }
    required_packages = ["torch"]
    models = [
        {
            "id": "Vchitect/VBench-2.0_models",
            "type": "huggingface",
            "task": "VBench 2.0 anatomy and identity checkpoints",
            "size": "2.26 GB",
            "auto_download": True,
        },
        {
            "id": MIRROR_REPO,
            "type": "huggingface",
            "task": "Mirrored VBench 2.0 source and external runtime artifacts",
            "auto_download": True,
            "notes": f"Revision {MIRROR_REVISION}",
        },
        {
            "id": "Vchitect/VBench@45e79ec14e69a2187202c675d2dbce1a71843d53",
            "type": "other",
            "url": "https://github.com/Vchitect/VBench/tree/45e79ec14e69a2187202c675d2dbce1a71843d53/VBench-2.0",
            "task": "VBench 2.0 evaluator and dimension-specific checkpoints",
            "notes": "Apache-2.0 source; install the upstream VBench-2.0 package",
        }
    ]
    metric_info = {
        **{
            field: f"VBench 2.0 {dimension.replace('_', ' ')} score"
            for dimension, field in DIMENSION_FIELDS.items()
        },
        "vbench2_creativity_score": "VBench 2.0 creativity aggregate",
        "vbench2_commonsense_score": "VBench 2.0 commonsense aggregate",
        "vbench2_controllability_score": "VBench 2.0 controllability aggregate",
        "vbench2_human_fidelity_score": "VBench 2.0 human-fidelity aggregate",
        "vbench2_physics_score": "VBench 2.0 physics aggregate",
        "vbench2_total_score": "Mean of the five VBench 2.0 category aggregates",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._backend: Optional[str] = None
        self._vbench_cls: Optional[Type[Any]] = None
        self._device: Any = "cpu"
        self._default_full_info: Optional[Path] = None

    @staticmethod
    def _merge_checkpoint_tree(source: Path, destination: Path) -> None:
        """Expose nested HF checkpoint files in VBench's flat cache layout."""
        if not source.is_dir():
            return
        destination.mkdir(parents=True, exist_ok=True)
        for item in source.rglob("*"):
            if not item.is_file():
                continue
            target = destination / item.relative_to(source)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)

    @classmethod
    def _prepare_checkpoint_layout(cls, checkpoint_root: Path) -> None:
        cls._merge_checkpoint_tree(
            checkpoint_root / "Human_Anatomy" / "YOLO-World",
            checkpoint_root / "YOLO-World",
        )
        cls._merge_checkpoint_tree(
            checkpoint_root / "Human_Anatomy" / "anomaly_detector",
            checkpoint_root / "anomaly_detector",
        )
        identity_root = checkpoint_root / "Human_Identity"
        arcface_source = identity_root / "arcface"
        cls._merge_checkpoint_tree(
            arcface_source if arcface_source.is_dir() else identity_root,
            checkpoint_root / "arcface",
        )

    @staticmethod
    def _extract_zip(archive: Path, destination: Path) -> Path:
        marker = destination / ".complete"
        if marker.is_file():
            roots = [path for path in destination.iterdir() if path.is_dir()]
            return roots[0] if len(roots) == 1 else destination
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (root / member.filename).resolve()
                try:
                    target.relative_to(root)
                except ValueError as exc:
                    raise ValueError(f"Unsafe VBench archive member: {member.filename}") from exc
            bundle.extractall(root)
        marker.touch()
        roots = [path for path in destination.iterdir() if path.is_dir()]
        return roots[0] if len(roots) == 1 else destination

    @staticmethod
    def _vendored_source() -> Path:
        """Path of the in-tree VBench 2.0 evaluator.

        Returns:
            Path: Directory placed on ``sys.path`` so ``import vbench2`` resolves
            to the vendored copy.

        Raises:
            RuntimeError: The vendored tree is missing.
        """
        from ayase import vendor

        root = Path(vendor.__file__).resolve().parent / "vbench"
        if not (root / "vbench2" / "__init__.py").is_file():
            raise RuntimeError("VBench 2.0 evaluator is missing from ayase.vendor")
        return root

    def _mirror_url(self, path: str) -> str:
        revision = quote(str(self.config.get("mirror_revision", MIRROR_REVISION)), safe="")
        return f"{MIRROR_RESOLVE_BASE}/{revision}/{path}"

    @staticmethod
    def _ensure_offline_gdown_compatibility() -> None:
        """Satisfy VBench's eager gdown import while disabling Drive fallbacks."""

        def unavailable(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("VBench fallback downloads are disabled; use the Ayase HF mirror")

        module = types.ModuleType("gdown")
        module.download = unavailable  # type: ignore[attr-defined]
        module.download_folder = unavailable  # type: ignore[attr-defined]
        sys.modules["gdown"] = module

    def _download_external_artifacts(self, checkpoint_root: Path, models_dir: str) -> Path:
        from ayase.config import download_hf_snapshot, download_model_file

        source_root = self._vendored_source()

        raft_archive = download_model_file(
            "vbench2/raft/models.zip",
            self._mirror_url("vbench2/raft/models.zip"),
            models_dir,
        )
        self._extract_zip(raft_archive, checkpoint_root / "raft_model")

        mirror_files = {
            "vbench2/arcface/resnet18_110.pth": "arcface/resnet18_110.pth",
            "vbench2/instance_anomaly_detector/model/adapter_config.json": "instance_anomaly_detector/model/adapter_config.json",
            "vbench2/instance_anomaly_detector/model/adapter_model.safetensors": "instance_anomaly_detector/model/adapter_model.safetensors",
            "vbench2/instance_anomaly_detector/model/additional_config.json": "instance_anomaly_detector/model/additional_config.json",
            "vbench2/instance_anomaly_detector/model/args.json": "instance_anomaly_detector/model/args.json",
        }
        for remote_path, cache_path in mirror_files.items():
            downloaded = download_model_file(
                f"vbench2/mirror/{remote_path}", self._mirror_url(remote_path), models_dir
            )
            target = checkpoint_root / cache_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(downloaded, target)
                except OSError:
                    shutil.copy2(downloaded, target)

        import torch

        hub_dir = Path(models_dir).resolve() / "torch_hub"
        torch.hub.set_dir(str(hub_dir))
        vgg19 = download_model_file(
            "vbench2/torchvision/vgg19-dcbb9e9d.pth",
            self._mirror_url("vbench2/torchvision/vgg19-dcbb9e9d.pth"),
            models_dir,
        )
        vgg19_cache = hub_dir / "checkpoints" / vgg19.name
        vgg19_cache.parent.mkdir(parents=True, exist_ok=True)
        if not vgg19_cache.exists():
            try:
                os.link(vgg19, vgg19_cache)
            except OSError:
                shutil.copy2(vgg19, vgg19_cache)
        retina = download_model_file(
            "vbench2/retinaface/retinaface_resnet50_2020-07-20-f168fae3c.zip",
            self._mirror_url(
                "vbench2/retinaface/retinaface_resnet50_2020-07-20-f168fae3c.zip"
            ),
            models_dir,
        )
        retina_cache = hub_dir / "checkpoints" / retina.name
        retina_cache.parent.mkdir(parents=True, exist_ok=True)
        if not retina_cache.exists():
            try:
                os.link(retina, retina_cache)
            except OSError:
                shutil.copy2(retina, retina_cache)

        cotracker_root = self._vendored_source() / "vbench2" / "third_party"
        self._merge_checkpoint_tree(cotracker_root, hub_dir / "facebookresearch_co-tracker_main")
        download_model_file(
            "torch_hub/checkpoints/cotracker2.pth",
            "https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth",
            models_dir,
        )

        dimensions = {str(value) for value in self.config.get("dimensions", DIMENSION_FIELDS)}
        llava_dimensions = {
            "Human_Clothes", "Composition", "Dynamic_Spatial_Relationship",
            "Dynamic_Attribute", "Motion_Rationality", "Mechanics", "Thermotics",
            "Material", "Complex_Landscape", "Complex_Plot", "Human_Interaction",
            "Motion_Order_Understanding",
        }
        if dimensions & llava_dimensions:
            llava_root = download_hf_snapshot("lmms-lab/LLaVA-Video-7B-Qwen2", models_dir)
            self._merge_checkpoint_tree(
                llava_root, checkpoint_root / "lmms-lab" / "LLaVA-Video-7B-Qwen2"
            )
        if dimensions & {
            "Complex_Landscape", "Complex_Plot", "Human_Interaction",
            "Motion_Order_Understanding",
        }:
            qwen_root = download_hf_snapshot("Qwen/Qwen2.5-7B-Instruct", models_dir)
            self._merge_checkpoint_tree(
                qwen_root, checkpoint_root / "Qwen" / "Qwen2.5-7B-Instruct"
            )
        return source_root

    def setup(self) -> None:
        try:
            from ayase.config import download_hf_snapshot

            models_dir = str(self.config.get("models_dir", "models"))
            checkpoint_root = download_hf_snapshot(
                str(self.config.get("model_repo", "Vchitect/VBench-2.0_models")),
                models_dir,
                revision=self.config.get("model_revision"),
            )
            self._prepare_checkpoint_layout(checkpoint_root)
            source_root = self._download_external_artifacts(checkpoint_root, models_dir)
            package_root = source_root
            if str(package_root) not in sys.path:
                sys.path.insert(0, str(package_root))
            # VBench 2.0 reads this at import time. Point it directly at the
            # snapshot, whose Human_Anatomy/Human_Identity layout matches the
            # upstream cache layout.
            os.environ["VBENCH2_CACHE_DIR"] = str(checkpoint_root)

            self._ensure_offline_gdown_compatibility()
            import torch
            import vbench2
            from vbench2 import VBench2
            from vbench2 import utils as vbench2_utils

            from ayase.runtime import resolve_torch_device

            self._device = torch.device(resolve_torch_device(self.config.get("device", "auto")))
            self._vbench_cls = VBench2
            self._default_full_info = package_root / "vbench2" / "VBench2_full_info.json"
            # Some installations imported utils before the environment variable
            # was set. Keep its module-level cache root synchronized, then ask
            # the upstream initializer to resolve every selected dependency now
            # instead of deferring downloads until post_process().
            if hasattr(vbench2_utils, "CACHE_DIR"):
                vbench2_utils.CACHE_DIR = str(checkpoint_root)
            dimensions = [str(value) for value in self.config.get("dimensions", DIMENSION_FIELDS)]
            vbench2_utils.init_submodules(
                dimensions,
                local=True,
                read_frame=bool(self.config.get("read_frame", False)),
            )
            self._backend = "vbench2"
        except Exception as exc:
            self._backend = "unavailable"
            logger.warning("VBench 2.0 unavailable: %s", exc)

    def process(self, sample: Sample) -> Sample:
        return sample

    @staticmethod
    def _extract_score(value: Any) -> Optional[float]:
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        if isinstance(value, dict):
            for key in ("score", "all_results", "overall"):
                if key in value:
                    value = value[key]
                    break
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @staticmethod
    def _aggregate_scores(scores: Dict[str, float]) -> Dict[str, float]:
        aggregates: Dict[str, float] = {}
        for field, dimensions in AGGREGATE_DIMENSIONS.items():
            if all(dimension in scores for dimension in dimensions):
                aggregates[field] = sum(scores[dimension] for dimension in dimensions) / len(
                    dimensions
                )
        category_fields = tuple(AGGREGATE_DIMENSIONS)
        if all(field in aggregates for field in category_fields):
            aggregates["vbench2_total_score"] = sum(
                aggregates[field] for field in category_fields
            ) / len(category_fields)
        return aggregates

    def _resolve_videos_path(self, samples: Sequence[Sample]) -> Optional[Path]:
        configured = self.config.get("videos_path")
        if configured:
            return Path(configured)
        parents = {sample.path.parent.resolve() for sample in samples if sample.is_video}
        if len(parents) == 1:
            return parents.pop()
        logger.warning(
            "VBench 2.0 requires one videos directory; set videos_path when samples span folders"
        )
        return None

    def post_process(self, all_samples: List[Sample]) -> None:
        if self._backend != "vbench2" or self._vbench_cls is None:
            return
        video_samples = [sample for sample in all_samples if sample.is_video]
        if not video_samples or self.pipeline is None:
            return

        dimensions = [str(value) for value in self.config.get("dimensions", DIMENSION_FIELDS)]
        unknown = sorted(set(dimensions) - set(DIMENSION_FIELDS))
        if unknown:
            logger.warning("Unknown VBench 2.0 dimensions: %s", ", ".join(unknown))
            return
        mode = str(self.config.get("mode", "vbench_standard"))
        if mode not in {"custom_input", "vbench_standard", "vbench_category"}:
            logger.warning("Unknown VBench 2.0 mode: %s", mode)
            return
        if mode == "custom_input":
            unsupported = sorted(set(dimensions) - CUSTOM_INPUT_DIMENSIONS)
            if unsupported:
                logger.warning(
                    "VBench 2.0 custom-input mode does not support: %s",
                    ", ".join(unsupported),
                )
                return

        videos_path = self._resolve_videos_path(video_samples)
        if videos_path is None:
            return
        full_info_path = self.config.get("full_info_path") or self._default_full_info
        if mode != "custom_input" and (
            not full_info_path or not Path(full_info_path).is_file()
        ):
            logger.warning("VBench 2.0 requires full_info_path to VBench2_full_info.json")
            return
        if full_info_path is None:
            full_info_path = ""

        output_dir = Path(self.config.get("output_dir", "reports/vbench2"))
        result_name = str(self.config.get("result_name", "ayase_vbench2"))
        try:
            evaluator = self._vbench_cls(self._device, str(full_info_path), str(output_dir))
            evaluator.evaluate(
                videos_path=str(videos_path),
                name=result_name,
                dimension_list=dimensions,
                local=True,
                read_frame=bool(self.config.get("read_frame", False)),
                mode=mode,
            )
            result_path = output_dir / f"{result_name}_eval_results.json"
            raw_results = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("VBench 2.0 evaluation failed: %s", exc)
            return

        scores: Dict[str, float] = {}
        for dimension, field in DIMENSION_FIELDS.items():
            score = self._extract_score(raw_results.get(dimension))
            if score is None:
                continue
            scores[dimension] = score
            self.pipeline.add_dataset_metric(field, score)
        for field, score in self._aggregate_scores(scores).items():
            self.pipeline.add_dataset_metric(field, score)

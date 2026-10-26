"""Morpheus physical-invariance and dynamical physics scores for annotated videos.

Runs the pinned author pipeline end to end: source-specific frame transformation,
SAM2.1 tracking from caller-supplied first-frame clicks, Depth Anything V1 lifting,
trajectory cropping, and the released physical/PINN scorers. Both scores are 0-1
and higher is better. Unsupported experiments or missing annotations are skipped.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Dict, Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

MORPHEUS_SHA = "ee3b6aec1474a9b905eba9a52efb5ea6cfd9f0bc"
SAM2_SHA = "412c70ccf2e9ff47798d052d93cd5aab1eb39351"
DEPTH_REVISION = "27ccb0920352c0c37b3a96441873c8d37bd52fb6"
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
)
_SOURCE_ARCHIVE = f"https://codeload.github.com/physics-from-video/Morpheus/zip/{MORPHEUS_SHA}"
_SAM2_ARCHIVE = f"https://codeload.github.com/physics-from-video/sam2_local/zip/{SAM2_SHA}"

_SUPPORTED_EXPERIMENTS = {
    "falling_ball_basketball", "falling_ball_baseball", "falling_ball_pingpong",
    "projectile_beach_volleyball", "projectile_kitchen_lemon", "projectile_pinecone",
    "falling_ball", "falling_apple", "falling_tape", "falling_marker", "projectile",
    "bouncing_ball", "sliding_book", "rolling_orange", "rolling_empty_can",
    "rolling_full_can", "holonomic_pendulum", "non_holonomic_pendulum",
    "double_pendulum", "collision_equal", "collision_small_hits_big",
    "collision_big_hits_small", "spring", "holonomic_pendulum_pendulum",
    "holonomic_pendulum_wrecking_ball", "rolling_full_can_aluminum_barrel",
    "rolling_full_can_barrel", "rolling_full_can_beverage_can", "sliding_book_brick",
    "sliding_book_delivery_truck_crate", "sliding_book_inclined_shelf",
}
_PREPROCESSING_MODELS = {
    "WAN-2.1", "Veo3-fast", "Veo3", "COSMOS-predict2", "COSMOS-predict1",
    "CogVideo", "PyramidalFlow", "LTX", "Kling-Turbo",
}
_PREPROCESSING_PATH_TOKENS = {"PyramidalFlow": "pyramid-flow"}
_TWO_OBJECT_EXPERIMENTS = {
    "holonomic_pendulum", "holonomic_pendulum_pendulum",
    "holonomic_pendulum_wrecking_ball", "double_pendulum",
    "collision_equal", "collision_small_hits_big",
    "collision_big_hits_small",
}
_LABEL_BUCKETS = {
    "falling_ball_basketball": "falling_ball", "falling_ball_baseball": "falling_ball",
    "falling_ball_pingpong": "falling_ball", "projectile_beach_volleyball": "projectile",
    "projectile_kitchen_lemon": "projectile", "projectile_pinecone": "projectile",
    "holonomic_pendulum_pendulum": "holonomic_pendulum",
    "holonomic_pendulum_wrecking_ball": "holonomic_pendulum",
    "rolling_full_can_aluminum_barrel": "rolling_full_can",
    "rolling_full_can_barrel": "rolling_full_can",
    "rolling_full_can_beverage_can": "rolling_full_can",
    "sliding_book_brick": "sliding_book", "sliding_book_delivery_truck_crate": "sliding_book",
    "sliding_book_inclined_shelf": "sliding_book",
    "non_holonomic_pendulum": "non_holonomic",
}


def _preprocessing_path_token(model: str) -> str:
    return _PREPROCESSING_PATH_TOKENS.get(model, model)


def _verify_source(
    root: Path, revision: str, required: tuple[str, ...], *, require_marker: bool = True
) -> None:
    marker = root / ".ayase-source-revision"
    if not all((root / item).is_file() for item in required):
        raise ValueError(f"Incomplete pinned source tree: {root}")
    if require_marker and not marker.is_file():
        raise ValueError(f"Unverified pinned source tree: {root}")
    if marker.is_file() and marker.read_text(encoding="ascii").strip() != revision:
        raise ValueError(f"Pinned source mismatch at {root}")


def _ensure_source(archive_url: str, revision: str, destination: Path, required: tuple[str, ...]) -> Path:
    """Safely acquire one immutable GitHub archive without requiring git."""
    from ayase.managed_runtime import _SetupLock

    destination.parent.mkdir(parents=True, exist_ok=True)
    with _SetupLock(destination.with_suffix(".lock")):
        if destination.is_dir():
            _verify_source(destination, revision, required)
            return destination
        with tempfile.TemporaryDirectory(prefix="morpheus-source-", dir=destination.parent) as name:
            temp = Path(name)
            archive = temp / "source.zip"
            urllib.request.urlretrieve(archive_url, archive)
            unpacked = temp / "unpacked"
            unpacked.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                members = {item.filename: item for item in bundle.infolist()}
                roots = {Path(item.filename).parts[0] for item in members.values() if Path(item.filename).parts}
                if len(roots) != 1:
                    raise ValueError("Pinned source archive must contain one root")
                symlinks = []
                for item in members.values():
                    target = (unpacked / item.filename).resolve()
                    if unpacked.resolve() not in target.parents and target != unpacked.resolve():
                        raise ValueError(f"Unsafe source archive member: {item.filename}")
                    if item.external_attr >> 16 & 0o170000 == 0o120000:
                        symlinks.append(item)
                    else:
                        bundle.extract(item, unpacked)
                for item in symlinks:
                    link = PurePosixPath(item.filename)
                    link_target = bundle.read(item).decode("utf-8")
                    resolved_member = str(link.parent.joinpath(link_target))
                    resolved_parts = []
                    for part in PurePosixPath(resolved_member).parts:
                        if part == "..":
                            if not resolved_parts:
                                raise ValueError(f"Unsafe source archive symlink: {item.filename}")
                            resolved_parts.pop()
                        elif part not in ("", "."):
                            resolved_parts.append(part)
                    resolved_member = "/".join(resolved_parts)
                    source_item = members.get(resolved_member)
                    if source_item is None or source_item.external_attr >> 16 & 0o170000 == 0o120000:
                        raise ValueError(f"Unsafe or chained source archive symlink: {item.filename}")
                    destination_file = unpacked / item.filename
                    destination_file.parent.mkdir(parents=True, exist_ok=True)
                    destination_file.write_bytes(bundle.read(source_item))
            extracted = unpacked / roots.pop()
            (extracted / ".ayase-source-revision").write_text(revision, encoding="ascii")
            _verify_source(extracted, revision, required)
            extracted.replace(destination)
    return destination


def _normalize_labels(labels: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(labels, dict):
        return None
    normalized: Dict[str, Any] = {}
    for key, value in labels.items():
        if not isinstance(key, str) or not key.rsplit("_", 1)[-1].isdigit():
            return None
        canonical = f"object_{int(key.rsplit('_', 1)[-1])}"
        if canonical in normalized:
            return None
        normalized[canonical] = value
    return normalized


def _valid_labels(labels: Any, experiment: str) -> bool:
    labels = _normalize_labels(labels)
    if labels is None:
        return False
    required = 2 if experiment in _TWO_OBJECT_EXPERIMENTS else 1
    if any(f"object_{index}" not in labels for index in range(1, required + 1)):
        return False
    for item in labels.values():
        if not isinstance(item, dict) or not item.get("positive"):
            return False
        for kind in ("positive", "negative"):
            points = item.get(kind, [])
            if not isinstance(points, list) or any(
                not isinstance(point, (list, tuple))
                or len(point) != 2
                or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
                for point in points
            ):
                return False
    return True


class MorpheusModule(PipelineModule):
    name = "morpheus"
    implementation_status = "implemented"
    implementation_notes = "Pins and automatically fetches actual upstream Morpheus/SAM2 source plus trained SAM2/depth assets; invokes author preprocessing/tracking/physical and PINN statistical scoring for explicit experiment annotations. Minor optional SAM2 postprocessing adaptation disclosed."
    implementation_review = "Source inspection; confidence medium; pretrained inference not validated."
    input_type = "video"
    description = "Morpheus native annotated-video physical and dynamical physics evaluation"
    provenance = {
        "morpheus_physical_score": "adapted",
        "morpheus_dynamical_score": "adapted",
    }
    sources = {
        "morpheus_physical_score": f"Morpheus source {MORPHEUS_SHA} — https://github.com/physics-from-video/Morpheus/tree/{MORPHEUS_SHA}",
        "morpheus_dynamical_score": f"Morpheus source {MORPHEUS_SHA} — https://github.com/physics-from-video/Morpheus/tree/{MORPHEUS_SHA}",
    }
    deviations = {
        "morpheus_physical_score": "SAM2 runs without its optional compiled _C connected-component/small-hole postprocessing; the upstream supported fallback leaves masks unchanged.",
        "morpheus_dynamical_score": "SAM2 runs without its optional compiled _C connected-component/small-hole postprocessing; the upstream supported fallback leaves masks unchanged.",
    }
    default_config = {
        "experiment": None,
        "object_labels": None,
        "preprocessing_model": None,
        "source_category": "generated",
        "conditioning": "single_frame_conditioning",
        "prompt_type": "plain",
        "models_dir": "models",
        "source_root": None,
        "sam2_source_root": None,
        "device": "auto",
    }
    models = [
        {"id": "sam2.1_hiera_large.pt", "type": "local", "url": SAM2_CHECKPOINT_URL, "task": "SAM2.1 Hiera-L checkpoint"},
        {"id": "nielsr/depth-anything-large", "type": "huggingface", "revision": DEPTH_REVISION, "auto_download": True, "task": "Depth Anything V1 normalized display-depth estimation"},
        {"id": "collision_mass_mlp.pt", "type": "local", "url": f"https://raw.githubusercontent.com/physics-from-video/Morpheus/{MORPHEUS_SHA}/morpheus/scoring/collision_mass_mlp.pt", "auto_download": True, "notes": "Acquired inside the pinned Morpheus source archive", "task": "Released Morpheus collision mass estimator"},
    ]
    metric_info = {
        "morpheus_physical_score": "Morpheus Physical Invariance Score (0-1, higher=better; author MAX-window recipes)",
        "morpheus_dynamical_score": "Morpheus Dynamical Score, 1-min(NMSE,1) from the author PINN/analytic scorer (0-1)",
    }
    metric_groups = {
        "morpheus_physical_score": "motion",
        "morpheus_dynamical_score": "motion",
    }
    implementation_sources = (
        "../third_party/morpheus/LICENSE",
        "../third_party/morpheus/NOTICE",
    )

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend = None
        self._tracker = None
        self._source_root: Optional[Path] = None

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            self._prepare_backend()
            self._backend = "morpheus-native"
        except Exception as exc:
            self._backend = "unavailable"
            logger.warning("Morpheus backend setup failed: %s", exc)

    def _applicable_config(self) -> bool:
        experiment = self.config.get("experiment")
        model = self.config.get("preprocessing_model")
        category = self.config.get("source_category")
        conditioning = self.config.get("conditioning")
        prompt_type = self.config.get("prompt_type")
        if experiment not in _SUPPORTED_EXPERIMENTS:
            logger.warning("Morpheus requires a supported explicit experiment; score left unset.")
            return False
        if model not in _PREPROCESSING_MODELS:
            logger.warning("Morpheus requires an explicit supported preprocessing_model; score left unset.")
            return False
        if category not in {"generated", "real-world"}:
            logger.warning("Morpheus source_category must be 'generated' or 'real-world'.")
            return False
        if conditioning not in {
            "single_frame_conditioning", "multi_frame_conditioning", "keyframe_interpolation"
        }:
            logger.warning("Morpheus requires a supported conditioning mode.")
            return False
        if prompt_type not in {"plain", "enhanced"}:
            logger.warning("Morpheus prompt_type must be 'plain' or 'enhanced'.")
            return False
        if not _valid_labels(self.config.get("object_labels"), experiment):
            logger.warning("Morpheus requires valid transformed-frame object labels; score left unset.")
            return False
        return True

    def _prepare_backend(self) -> None:
        from ayase.config import download_model_file
        from ayase.runtime import resolve_torch_device
        from huggingface_hub import snapshot_download

        models_dir = Path(self.config.get("models_dir", "models")).resolve()
        root = self.config.get("source_root")
        sam2_root = self.config.get("sam2_source_root")
        self._source_root = Path(root).resolve() if root else _ensure_source(
            _SOURCE_ARCHIVE, MORPHEUS_SHA, models_dir / ".ayase" / "sources" / f"morpheus-{MORPHEUS_SHA}",
            ("morpheus/pipeline.py", "morpheus/scoring/combined.py", "configs/sam2.1_hiera_l.yaml"),
        )
        sam2_path = Path(sam2_root).resolve() if sam2_root else _ensure_source(
            _SAM2_ARCHIVE, SAM2_SHA, models_dir / ".ayase" / "sources" / f"sam2-local-{SAM2_SHA}",
            ("sam2/__init__.py", "sam2/build_sam.py"),
        )
        _verify_source(
            self._source_root, MORPHEUS_SHA,
            ("morpheus/pipeline.py", "configs/sam2.1_hiera_l.yaml"),
            require_marker=not bool(root),
        )
        _verify_source(
            sam2_path, SAM2_SHA, ("sam2/__init__.py", "sam2/build_sam.py"),
            require_marker=not bool(sam2_root),
        )
        checkpoint = download_model_file(
            "morpheus/sam2.1_hiera_large.pt", SAM2_CHECKPOINT_URL, str(models_dir)
        )
        depth_path = Path(snapshot_download(
            repo_id="nielsr/depth-anything-large", revision=DEPTH_REVISION,
            local_dir=str(models_dir / "morpheus" / "depth-anything-large"),
        )).resolve()

        # Insert SAM2 last so its repository root has precedence over the
        # uninitialised ``sam2`` gitlink directory in the Morpheus archive.
        for path in (self._source_root, sam2_path):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import morpheus.tracking.processing as processing
        import morpheus as morpheus_package
        import sam2 as sam2_package
        if self._source_root not in Path(morpheus_package.__file__).resolve().parents:
            raise ImportError("Imported Morpheus does not originate from the pinned source root")
        if sam2_path not in Path(sam2_package.__file__).resolve().parents:
            raise ImportError("Imported SAM2 does not originate from the pinned source root")
        from morpheus.tracking.sam2_tracker import Sam2Tracker
        from transformers import pipeline as transformers_pipeline

        original_pipeline = processing.pipeline

        def pinned_pipeline(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("model") == "nielsr/depth-anything-large":
                kwargs["model"] = str(depth_path)
            return transformers_pipeline(*args, **kwargs)

        processing.pipeline = pinned_pipeline
        try:
            self._tracker = Sam2Tracker(
                checkpoint=str(checkpoint), config_dir=str(self._source_root / "configs"),
                model_cfg="sam2.1_hiera_l.yaml",
                device=resolve_torch_device(str(self.config.get("device", "auto"))), load_depth=True,
            )
        finally:
            processing.pipeline = original_pipeline

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or self._backend != "morpheus-native" or self._tracker is None:
            return sample
        try:
            annotations = dict(self.config)
            sample_annotations = sample.metadata.get("morpheus_annotations")
            if isinstance(sample_annotations, dict):
                annotations.update(sample_annotations)
            if not self._applicable_annotations(annotations):
                return sample
            scores, details = self._run_source_pipeline(sample.path, annotations)
            physical = scores.get("physical_score")
            dynamical = scores.get("statistical_score")
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            if isinstance(physical, (int, float)) and math.isfinite(physical):
                sample.quality_metrics.morpheus_physical_score = float(physical)
            if isinstance(dynamical, (int, float)) and math.isfinite(dynamical):
                sample.quality_metrics.morpheus_dynamical_score = float(dynamical)
            sample.metadata["morpheus"] = details
        except Exception as exc:
            logger.warning("Morpheus failed for %s: %s", sample.path, exc)
        return sample

    def on_dispose(self) -> None:
        self._tracker = None
        self._backend = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _applicable_annotations(self, annotations: Dict[str, Any]) -> bool:
        original = self.config
        try:
            self.config = annotations
            return self._applicable_config()
        finally:
            self.config = original

    def _run_source_pipeline(self, video_path: Path, annotations: Optional[Dict[str, Any]] = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
        import tempfile
        from morpheus.pipeline import process_video_folder
        from morpheus.resizing.transforms import process_mp4_file

        config = annotations or self.config
        experiment = str(config["experiment"])
        preprocessing_model = str(config["preprocessing_model"])
        preprocessing_path_token = _preprocessing_path_token(preprocessing_model)
        # The author scorer uses membership in MODEL_NAMES as its category switch.
        # A real-world request is represented explicitly after preprocessing.
        scorer_model = preprocessing_model if config["source_category"] == "generated" else "real-world"
        rel_video = os.path.join(
            str(config.get("conditioning", "single_frame_conditioning")),
            preprocessing_path_token, str(config.get("prompt_type", "plain")), experiment,
            video_path.name,
        )
        with tempfile.TemporaryDirectory(prefix="ayase-morpheus-") as tmp:
            work = Path(tmp)
            prepared_root, output_root = work / "prepared", work / "output"
            process_mp4_file(str(video_path), rel_video, str(prepared_root), video_num=0, use_cache=False)
            info_paths = list(prepared_root.rglob("info.json"))
            if len(info_paths) != 1:
                raise RuntimeError("Morpheus source preprocessing produced no video")
            info_path = info_paths[0]
            prepared = info_path.parent
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["experiment"] = experiment
            info["model"] = scorer_model
            if scorer_model == "real-world":
                info["conditioning"] = "real-world"
            info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
            if scorer_model == "real-world":
                real_world_prepared = prepared_root / "real-world" / experiment / "video_0"
                real_world_prepared.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(prepared), str(real_world_prepared))
                prepared = real_world_prepared
                info_path = prepared / "info.json"
            holonomic_reference = (
                experiment == "holonomic_pendulum"
                and scorer_model != "real-world"
                and config.get("conditioning") != "keyframe_interpolation"
            )
            label_key = "video_5" if holonomic_reference else "video_0"
            label_bucket = _LABEL_BUCKETS.get(experiment, experiment)
            normalized_labels = _normalize_labels(config["object_labels"])
            if normalized_labels is None:
                raise ValueError("invalid Morpheus object labels")
            labels = {label_bucket: {label_key: normalized_labels}}
            args = SimpleNamespace(
                input_dir=str(prepared_root), output_dir=str(output_root), video_number=None,
                use_cache=False, use_cached_tracking=False, save_segmentation_masks=False,
                save_depth_masks_v2=False, depth_processor_v2=None, calculate_scores=True,
                only_physical_score=False, only_statistical_score=False,
                resume_statistical_score=False,
            )
            process_video_folder(str(prepared), labels, self._tracker, args)
            score_path = output_root / prepared.relative_to(prepared_root) / "combined_scores.json"
            if not score_path.exists():
                raise RuntimeError("Morpheus source pipeline produced no scores")
            scores = json.loads(score_path.read_text(encoding="utf-8"))
            details = {
                "source_revision": MORPHEUS_SHA,
                "sam2_revision": SAM2_SHA,
                "depth_revision": DEPTH_REVISION,
                "experiment": experiment,
                "source_category": config["source_category"],
                "preprocessing_model": preprocessing_model,
                "components": {
                    "physical": scores.get("individual_physical_scores"),
                    "dynamical": scores.get("individual_statistical_scores"),
                },
                "dynamical_recipe": "author PINN/analytic recipe (200000 epochs at 1e-3 except source family overrides)",
                "physical_window_aggregation": "author MAX-window recipes",
                "sam2_connected_components": False,
                "sam2_postprocessing_note": "Optional SAM2 _C small-hole/connected-component adaptation unavailable in no-extension runtime; source fallbacks leave masks unchanged.",
            }
            return scores, details

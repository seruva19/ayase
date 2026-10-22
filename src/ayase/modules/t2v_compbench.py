"""Runner for the seven official T2V-CompBench category evaluators.

This adapter invokes a user-prepared checkout of the official repository. It
does not replace any learned evaluator with an Ayase approximation.
"""

import csv
import ast
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_REVISION = "dd5eff7b93af0550b9efa2bdabbb21b3b017ceda"
_CATEGORIES = (
    "consistent_attribute_binding",
    "dynamic_attribute_binding",
    "spatial_relationships",
    "motion_binding",
    "action_binding",
    "object_interactions",
    "generative_numeracy",
)
_ALIASES = {
    "consistent_attr": "consistent_attribute_binding",
    "dynamic_attr": "dynamic_attribute_binding",
    "spatial": "spatial_relationships",
    "motion": "motion_binding",
    "action": "action_binding",
    "interaction": "object_interactions",
    "numeracy": "generative_numeracy",
    **{name: name for name in _CATEGORIES},
}
_OUTPUTS = {
    "consistent_attribute_binding": "ayase_consistent_attr_score.csv",
    "dynamic_attribute_binding": "ayase_dynamic_attr_score.csv",
    "spatial_relationships": "ayase_spatial_score.csv",
    "motion_binding": "ayase_motion_score.csv",
    "action_binding": "ayase_action_binding_score.csv",
    "object_interactions": "ayase_object_interactions_score.csv",
    "generative_numeracy": "ayase_numeracy_video.csv",
}
_RAW_NORMALIZATION = {
    "consistent_attribute_binding": (1.0, 14.0),
    "action_binding": (1.0, 9.0),
    "object_interactions": (1.0, 9.0),
}
_PINNED_BLOBS = {
    "Depth-Anything/compbench_run_depth.py": "fcb535f4850edb644bffde0f0e2c8a877ca29d31",
    "Grounded-Segment-Anything/GroundingDINO/demo/compbench_eval_numeracy.py": "2bf9f668ab81994812c18e4cefa83ef54f435abc",
    "Grounded-Segment-Anything/compbench_eval_spatial_relationships.py": "a6d71fa61a36bbd783d40e07e443fb0c2c81d365",
    "Grounded-Segment-Anything/compbench_motion_binding_seg.py": "4a8caa2233c1227daf049617ae14b8f0fd01b7bc",
    "LLaVA/llava/eval/compbench_eval_action_binding.py": "4e3f38167877b88144a286b7c75b9a0c9913c4e9",
    "LLaVA/llava/eval/compbench_eval_consistent_attr.py": "30006a67f435df46b47e9ecbcf6c2cb84e038360",
    "LLaVA/llava/eval/compbench_eval_dynamic_attr.py": "6764278b57972cee21d0558a21a1f646786ab894",
    "LLaVA/llava/eval/compbench_eval_interaction.py": "f824ea47c2386d78aed9271a0c0ed92507f6b9dc",
    "dot/compbench_eval_motion_binding.py": "7214a3db689bd72a3b58da88ec21d20dbc76deb8",
    "dot/dot/utils/options/compbench_demo_options.py": "9bc5172f83f04d1ed896d3dada6ee30646114af1",
    "meta_data/action_binding.json": "515db9bac3deffb1264700e2b7003c74e21a8b99",
    "meta_data/consistent_attribute_binding.json": "1efc395dd4e608842fc1a75046b217eefe1cb94e",
    "meta_data/dynamic_attribute_binding.json": "6459bda1d00a19ab9e1655eaf82cd136ee29bdfd",
    "meta_data/generative_numeracy.json": "e33e52c017a996f6aedf735d213eea10f8c924f5",
    "meta_data/motion_binding.json": "2fc50f83876decf33aa50404937474830032b527",
    "meta_data/object_interactions.json": "440140890487a219429da8c873ecf068e5aa4b17",
    "meta_data/spatial_relationships.json": "a6a90d4010ebff540b494af42feda87c3b2f8ea7",
}


class T2VCompBenchOfficialModule(PipelineModule):
    name = "t2v_compbench_official"
    provenance = "adapted"
    input_type = "video"
    description = "Official seven-category T2V-CompBench evaluator runner"
    metric_groups = {
        f"t2v_compbench_{category}": "alignment" for category in _CATEGORIES
    }
    deviations = {
        f"t2v_compbench_{category}": (
            "Ayase stages selected benchmark IDs rather than requiring all 200; "
            "the upstream evaluator and score formula are unchanged"
        )
        for category in _CATEGORIES
    }
    deviations.update(
        {
            f"t2v_compbench_{category}_avg": (
                "Mean over the exact staged subset; upstream score formula is unchanged"
            )
            for category in _CATEGORIES
        }
    )
    sources = {
        f"t2v_compbench_{category}": (
            "Official T2V-CompBench evaluator at " + _REVISION
            + " — https://github.com/KaiyueSun98/T2V-CompBench"
        )
        for category in _CATEGORIES
    }
    sources.update(
        {
            f"t2v_compbench_{category}_avg": (
                "Official T2V-CompBench evaluator at " + _REVISION
                + " — https://github.com/KaiyueSun98/T2V-CompBench"
            )
            for category in _CATEGORIES
        }
    )
    default_config = {
        "upstream_root": None,
        "metadata_dir": None,
        "categories": list(_CATEGORIES),
        "category": None,
        "python_executables": {},
        "model_name": "ayase",
        "timeout": 7200,
        "temp_root": None,
        "total_frames": 16,
        "fps": 8,
        "strict_prompt_match": True,
        "verify_revision": True,
        "check_fixed_assets": True,
    }
    models = [
        {
            "id": "KaiyueSun98/T2V-CompBench@" + _REVISION,
            "type": "other",
            "url": "https://github.com/KaiyueSun98/T2V-CompBench",
            "task": "Official LLaVA, GroundingDINO/SAM/Depth Anything, and DOT evaluators",
            "notes": "User-prepared checkout and the evaluator-specific environments are required",
        }
    ]
    metric_info = {
        f"t2v_compbench_{category}": f"Official {category.replace('_', ' ')} per-video score"
        for category in _CATEGORIES
    }
    metric_info.update(
        {
            f"t2v_compbench_{category}_avg": (
                f"Mean official {category.replace('_', ' ')} score over evaluated videos"
            )
            for category in _CATEGORIES
        }
    )

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        selected = self.config.get("category") or self.config.get("categories", _CATEGORIES)
        if isinstance(selected, str):
            selected = [selected]
        self.categories = tuple(self._canonical_category(item) for item in selected)
        self._samples: Dict[str, Dict[int, Sample]] = {name: {} for name in self.categories}
        self._invalid_categories = set()
        self._runner = subprocess.run
        self._available = False
        self._backend = None

    @staticmethod
    def _canonical_category(value: str) -> str:
        try:
            return _ALIASES[str(value).lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported T2V-CompBench category: {value}") from exc

    def setup(self) -> None:
        if self.test_mode:
            self._available = False
            self._backend = None
            return
        self._available = False
        self._backend = "unavailable"
        root = self._root()
        if root is None:
            logger.warning("t2v_compbench_official requires config upstream_root")
            return
        missing = [spec[0] for name in self.categories for spec in self._command_specs(name) if not (root / spec[0]).is_file()]
        if missing:
            logger.warning("T2V-CompBench checkout is incomplete; missing %s", sorted(set(missing)))
            return
        try:
            if self.config.get("verify_revision", True):
                self._verify_pinned_sources(root)
            for category in self.categories:
                self._load_metadata(root, category)
                self._validate_cli_contract(root, category)
            if self.config.get("check_fixed_assets", True):
                self._check_fixed_assets(root)
        except ValueError as exc:
            logger.warning("T2V-CompBench checkout validation failed: %s", exc)
            return
        self._available = True
        self._backend = "official_t2v_compbench"

    def _check_fixed_assets(self, root: Path) -> None:
        required = set()
        detection = {"spatial_relationships", "generative_numeracy", "motion_binding"}
        if detection.intersection(self.categories):
            required.add(
                "Grounded-Segment-Anything/GroundingDINO/weights/groundingdino_swint_ogc.pth"
            )
        if {"spatial_relationships", "motion_binding"}.intersection(self.categories):
            required.add("Grounded-Segment-Anything/sam_vit_h_4b8939.pth")
        if "motion_binding" in self.categories:
            required.update(
                {
                    "dot/checkpoints/cvo_raft_patch_8.pth",
                    "dot/checkpoints/movi_f_raft_patch_4_alpha.pth",
                    "dot/checkpoints/movi_f_cotracker2_patch_4_wind_8.pth",
                }
            )
        missing = [
            str(root / relative)
            for relative in sorted(required)
            if not (root / relative).is_file()
        ]
        if missing:
            raise ValueError("required evaluator assets are missing: " + ", ".join(missing))

    @staticmethod
    def _git_blob_hash(path: Path) -> str:
        payload = path.read_bytes()
        # Git's text checkout may materialize CRLF on Windows while the pinned
        # blob is stored with LF. Match Git's normalized source identity.
        payload = payload.replace(b"\r\n", b"\n")
        header = f"blob {len(payload)}\0".encode("ascii")
        return hashlib.sha1(header + payload).hexdigest()  # noqa: S324 - Git object identity

    def _verify_pinned_sources(self, root: Path) -> None:
        metadata_root = self._metadata_dir(root)
        for relative, expected in _PINNED_BLOBS.items():
            if relative.startswith("meta_data/"):
                path = metadata_root / Path(relative).name
            else:
                path = root / relative
            if not path.is_file() or self._git_blob_hash(path) != expected:
                raise ValueError(f"source differs from pinned revision {_REVISION}: {path}")

    @staticmethod
    def _declared_options(path: Path) -> set:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        options = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "add_argument":
                continue
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if argument.value.startswith("--"):
                        options.add(argument.value)
        return options

    def _validate_cli_contract(self, root: Path, category: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            commands = self._commands(root, category, Path(directory))
        specs = self._command_specs(category)
        for index, (command, _cwd) in enumerate(commands):
            parser_path = root / specs[index][0]
            if category == "motion_binding" and index == 1:
                parser_path = root / "dot/dot/utils/options/compbench_demo_options.py"
            declared = self._declared_options(parser_path)
            supplied = {item for item in command[2:] if item.startswith("--")}
            unknown = supplied - declared
            if unknown:
                raise ValueError(f"unsupported arguments for {category}: {sorted(unknown)}")

    def _root(self) -> Optional[Path]:
        value = self.config.get("upstream_root")
        if not value:
            return None
        root = Path(value).resolve()
        return root if root.is_dir() else None

    def _metadata_dir(self, root: Path) -> Path:
        configured = self.config.get("metadata_dir")
        return Path(configured).resolve() if configured else root / "meta_data"

    def _sample_category(self, sample: Sample) -> Optional[str]:
        if len(self.categories) == 1:
            return self.categories[0]
        for part in reversed(Path(sample.path).parts[:-1]):
            category = _ALIASES.get(part.lower())
            if category in self.categories:
                return category
        return None

    @staticmethod
    def _benchmark_id(path: Path) -> Optional[int]:
        stem = path.stem
        if len(stem) < 4 or not stem[:4].isdigit() or (len(stem) > 4 and stem[4].isdigit()):
            return None
        value = int(stem[:4])
        return value if 1 <= value <= 200 else None

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample
        category = self._sample_category(sample)
        identifier = self._benchmark_id(Path(sample.path))
        if category is None or identifier is None or not Path(sample.path).is_file():
            return sample
        existing = self._samples[category].get(identifier)
        if existing is not None and Path(existing.path).resolve() != Path(sample.path).resolve():
            logger.warning(
                "T2V-CompBench duplicate id %04d for %s; category rejected",
                identifier,
                category,
            )
            self._samples[category].pop(identifier, None)
            self._invalid_categories.add(category)
            return sample
        if category not in self._invalid_categories:
            self._samples[category][identifier] = sample
        return sample

    def _load_metadata(self, root: Path, category: str) -> List[dict]:
        path = self._metadata_dir(root) / f"{category}.json"
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list) or len(records) != 200:
            raise ValueError(f"invalid benchmark metadata: {path}")
        return records

    def _validated_samples(self, root: Path, category: str) -> List[Tuple[int, Sample]]:
        metadata = self._load_metadata(root, category)
        accepted = []
        for identifier, sample in sorted(self._samples[category].items()):
            expected = str(metadata[identifier - 1].get("prompt", "")).strip()
            caption = getattr(getattr(sample, "caption", None), "text", None)
            if self.config.get("strict_prompt_match", True) and caption and caption.strip() != expected:
                logger.warning("T2V-CompBench prompt mismatch for %s id %04d", category, identifier)
                continue
            accepted.append((identifier, sample))
        return accepted

    @staticmethod
    def _stage(samples: Sequence[Tuple[int, Sample]], destination: Path) -> None:
        destination.mkdir(parents=True)
        for identifier, sample in samples:
            target = destination / f"{identifier:04d}.mp4"
            try:
                os.link(Path(sample.path), target)
            except OSError:
                shutil.copy2(Path(sample.path), target)

    def _python(self, category: str) -> str:
        configured = self.config.get("python_executables", {})
        return str(configured.get(category, configured.get("default", sys.executable)))

    @staticmethod
    def _command_specs(category: str) -> List[Tuple[str, str]]:
        if category == "consistent_attribute_binding":
            return [("LLaVA/llava/eval/compbench_eval_consistent_attr.py", "root")]
        if category == "dynamic_attribute_binding":
            return [("LLaVA/llava/eval/compbench_eval_dynamic_attr.py", "root")]
        if category == "action_binding":
            return [("LLaVA/llava/eval/compbench_eval_action_binding.py", "root")]
        if category == "object_interactions":
            return [("LLaVA/llava/eval/compbench_eval_interaction.py", "root")]
        if category == "spatial_relationships":
            return [("Grounded-Segment-Anything/compbench_eval_spatial_relationships.py", "root")]
        if category == "generative_numeracy":
            return [("Grounded-Segment-Anything/GroundingDINO/demo/compbench_eval_numeracy.py", "root")]
        return [
            ("Grounded-Segment-Anything/compbench_motion_binding_seg.py", "root"),
            ("dot/compbench_eval_motion_binding.py", "dot"),
        ]

    def _commands(self, root: Path, category: str, run_dir: Path) -> List[Tuple[List[str], Path]]:
        python = self._python(category)
        video = run_dir / "videos" / category
        output = run_dir / "csv"
        artifacts = run_dir / "artifacts"
        metadata = self._metadata_dir(root) / f"{category}.json"
        common = ["--video-path", str(video), "--read-prompt-file", str(metadata)]
        model = str(self.config["model_name"])
        scripts = self._command_specs(category)
        if category in {"consistent_attribute_binding", "dynamic_attribute_binding", "action_binding", "object_interactions"}:
            return [([python, str(root / scripts[0][0]), *common, "--output-path", str(output), "--t2v-model", model], root)]
        if category == "spatial_relationships":
            command = [python, str(root / scripts[0][0]), *common, "--depth_folder", str(run_dir / "depth"), "--output-path", str(output), "--t2v-model", model, "--output_dir", str(artifacts)]
            return [(command, root)]
        if category == "generative_numeracy":
            command = [python, str(root / scripts[0][0]), *common, "--output-path", str(output), "--t2v-model", model, "--output_dir", str(artifacts)]
            return [(command, root)]
        mask_dir = run_dir / "masks"
        first = [python, str(root / scripts[0][0]), *common, "--t2v-model", model, "--total_frame", str(self.config["total_frames"]), "--fps", str(self.config["fps"]), "--output_dir", str(mask_dir)]
        standard = video.parent / "video_standard" / video.name
        second = [python, str(root / scripts[1][0]), "--video-path", str(standard), "--mask_folder", str(mask_dir), "--read-prompt-file", str(metadata), "--t2v-model", model, "--output-path", str(output), "--output_dir", str(artifacts)]
        return [(first, root), (second, root / "dot")]

    def _run_category(self, root: Path, category: str, run_dir: Path) -> Path:
        output = run_dir / "csv"
        output.mkdir(parents=True)
        for index, (command, cwd) in enumerate(self._commands(root, category, run_dir)):
            log_path = run_dir / f"evaluator-{index}.log"
            try:
                with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
                    self._runner(
                        command,
                        cwd=str(cwd),
                        shell=False,
                        check=True,
                        timeout=float(self.config["timeout"]),
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
            except (OSError, subprocess.SubprocessError) as exc:
                tail = b""
                try:
                    with log_path.open("rb") as log_file:
                        log_file.seek(0, os.SEEK_END)
                        size = log_file.tell()
                        log_file.seek(max(0, size - 2048))
                        tail = log_file.read(2048)
                except OSError:
                    pass
                detail = tail.decode("utf-8", errors="replace").strip()
                if detail:
                    raise RuntimeError(
                        f"official evaluator command failed: {exc}; log tail: {detail}"
                    ) from exc
                raise RuntimeError(f"official evaluator command failed: {exc}") from exc
        return output / _OUTPUTS[category].replace("ayase", str(self.config["model_name"]), 1)

    @staticmethod
    def _row_id(value: str) -> Optional[int]:
        name = Path(value).name
        return int(name[:4]) if len(name) >= 4 and name[:4].isdigit() else None

    @classmethod
    def _parse_scores(cls, path: Path, category: str, metadata: Sequence[dict]) -> Tuple[Dict[int, float], float]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if len(rows) < 3:
            raise ValueError(f"malformed T2V-CompBench output: {path}")
        header = rows[0]
        prompt_index = next((i for i, value in enumerate(header) if value.lower() == "prompt"), None)
        scores: Dict[int, float] = {}
        trailer = None
        for row in rows[1:]:
            if row and row[0].strip().lower().startswith("score:"):
                if len(row) > 1:
                    trailer = float(row[1])
                continue
            if len(row) < 2:
                continue
            identifier = cls._row_id(row[0])
            if identifier is None or not 1 <= identifier <= len(metadata):
                continue
            if prompt_index is not None and prompt_index < len(row):
                if row[prompt_index].strip() != str(metadata[identifier - 1].get("prompt", "")).strip():
                    raise ValueError(f"output prompt does not match benchmark id {identifier:04d}")
            try:
                value = float(row[-1])
            except ValueError:
                continue
            if category in _RAW_NORMALIZATION:
                offset, divisor = _RAW_NORMALIZATION[category]
                value = (value - offset) / divisor
            if not math.isfinite(value):
                raise ValueError(f"non-finite score for benchmark id {identifier:04d}")
            if identifier in scores:
                raise ValueError(f"duplicate score for benchmark id {identifier:04d}")
            scores[identifier] = value
        if not scores or trailer is None or not math.isfinite(trailer):
            raise ValueError(f"missing per-video scores or final score: {path}")
        mean_score = sum(scores.values()) / len(scores)
        if not math.isclose(mean_score, trailer, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(
                f"final score {trailer} does not match per-video mean {mean_score}"
            )
        return scores, trailer

    def post_process(self, all_samples: List[Sample]) -> None:
        del all_samples
        root = self._root()
        if not self._available or root is None:
            return
        configured_temp = self.config.get("temp_root")
        temp_root = Path(configured_temp) if configured_temp else None
        if temp_root is not None:
            temp_root.mkdir(parents=True, exist_ok=True)
        for category in self.categories:
            try:
                if category in self._invalid_categories:
                    logger.warning(
                        "T2V-CompBench %s skipped because input IDs are ambiguous",
                        category,
                    )
                    continue
                samples = self._validated_samples(root, category)
                if not samples:
                    continue
                with tempfile.TemporaryDirectory(prefix=f"{category}-", dir=temp_root) as directory:
                    run_dir = Path(directory)
                    self._stage(samples, run_dir / "videos" / category)
                    output = self._run_category(root, category, run_dir)
                    metadata = self._load_metadata(root, category)
                    scores, average = self._parse_scores(output, category, metadata)
                    expected_ids = {identifier for identifier, _sample in samples}
                    if set(scores) != expected_ids:
                        missing = sorted(expected_ids - set(scores))
                        extra = sorted(set(scores) - expected_ids)
                        raise ValueError(
                            f"output IDs differ from staged IDs; missing={missing}, extra={extra}"
                        )
                    field = f"t2v_compbench_{category}"
                    provenance = type(self).field_provenance().get(field, "adapted")
                    for identifier, sample in samples:
                        if identifier in scores and field in QualityMetrics.model_fields:
                            if sample.quality_metrics is None:
                                sample.quality_metrics = QualityMetrics()
                            setattr(sample.quality_metrics, field, scores[identifier])
                            sample.quality_metrics.metric_provenance[field] = provenance
                            if self._backend:
                                sample.quality_metrics.metric_backends[self.name] = self._backend
                    if self.pipeline is not None:
                        self.pipeline.add_dataset_metric(field + "_avg", average)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                logger.warning("T2V-CompBench %s failed: %s", category, exc)

    def on_dispose(self) -> None:
        for samples in self._samples.values():
            samples.clear()
        self._invalid_categories.clear()
        self._available = False
        self._backend = None
        super().on_dispose()

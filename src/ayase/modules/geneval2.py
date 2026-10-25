"""Image-prompt alignment using GenEval2's published Soft-TIFA protocol.

Per-image scores are raw sums of full-vocabulary first-token probabilities;
upstream deliberately counts duplicate token IDs, so values are not clamped. Full
benchmark aggregates are emitted only for one successful image per official prompt.
"""

import json
import hashlib
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

from ._geneval2_protocol import aggregate_atom_scores, score_question

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
BENCHMARK_SHA256 = "09233adbc9f877fba205f1a20efa4a8640e7e62a91dfd55893562bed8a928e43"
BENCHMARK_PROMPTS = 800


class GenEval2Module(PipelineModule):
    name = "geneval2"
    description = "GenEval2 Soft-TIFA image-prompt alignment (raw AM and GM)"
    input_type = "image"
    provenance = {
        "soft_tifa_am_score": "published",
        "soft_tifa_gm_score": "published",
        "geneval2_am_score": "published",
        "geneval2_gm_score": "published",
        "geneval2_coverage": "utility",
    }
    sources = {
        "soft_tifa_am_score": "GenEval2 Soft-TIFA — https://github.com/facebookresearch/GenEval2",
        "soft_tifa_gm_score": "GenEval2 Soft-TIFA — https://github.com/facebookresearch/GenEval2",
        "geneval2_am_score": "GenEval2 full-benchmark Soft-TIFA AM — https://github.com/facebookresearch/GenEval2",
        "geneval2_gm_score": "GenEval2 full-benchmark Soft-TIFA GM — https://github.com/facebookresearch/GenEval2",
        "geneval2_coverage": "Ayase completion diagnostic for https://github.com/facebookresearch/GenEval2",
    }
    implementation_sources = (
        "../third_party/geneval2/evaluation.py.source",
        "../third_party/geneval2/soft_tifa_analysis.py.source",
        "../third_party/geneval2/geneval2_data.jsonl",
        "_geneval2_protocol.py",
    )
    default_config = {
        "model_path": None,
        "benchmark_path": None,
        "models_dir": "models",
        "device": "auto",
        "dtype": "auto",
        "device_map": "auto",
    }
    models = [
        {
            "id": MODEL_ID,
            "type": "huggingface",
            "revision": MODEL_REVISION,
            "task": "GenEval2 Soft-TIFA visual question answering",
            "auto_download": True,
            "notes": "Apache-2.0 model; GenEval2 code/data are CC-BY-NC-4.0",
        }
    ]
    metric_info = {
        "soft_tifa_am_score": "Per-prompt arithmetic mean of atomic answer masses (raw)",
        "soft_tifa_gm_score": "Per-prompt geometric mean of atomic answer masses (raw)",
        "geneval2_am_score": "Official 800-prompt benchmark mean Soft-TIFA AM ×100",
        "geneval2_gm_score": "Official 800-prompt benchmark mean Soft-TIFA GM ×100",
        "geneval2_coverage": "Unique successfully scored official prompts / benchmark prompts",
    }
    metric_groups = {
        "soft_tifa_am_score": "alignment",
        "soft_tifa_gm_score": "alignment",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.models_dir = self.config.get("models_dir", "models")
        self._backend: Optional[str] = None
        self._processor = None
        self._model = None
        self._torch = None
        self._benchmark, self._benchmark_order, self._benchmark_valid = self._load_benchmark()
        self.config["_geneval2_benchmark_sha256"] = (
            self._benchmark_digest if self._benchmark_valid else None
        )
        self._canonical_benchmark = self._benchmark_valid and self._is_canonical_benchmark()

    def _load_benchmark(self) -> Tuple[Dict[str, Dict[str, Any]], List[str], bool]:
        configured = self.config.get("benchmark_path")
        path = Path(configured) if configured else (
            Path(__file__).resolve().parent.parent / "third_party" / "geneval2" / "geneval2_data.jsonl"
        )
        records: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        try:
            raw = path.read_bytes()
            normalized = raw.replace(b"\r\n", b"\n")
            for line in normalized.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("benchmark records must be JSON objects")
                prompt = record["prompt"]
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError("benchmark prompt must be a non-empty string")
                vqa_list = self._validate_vqa_list(record.get("vqa_list"), custom=False)
                if vqa_list is None:
                    raise ValueError(f"invalid VQA list for prompt: {prompt!r}")
                atom_count = record.get("atom_count")
                skills = record.get("skills")
                if not isinstance(atom_count, int) or atom_count < 1:
                    raise ValueError(f"invalid atom count for prompt: {prompt!r}")
                if not isinstance(skills, list) or len(skills) != len(vqa_list):
                    raise ValueError(f"invalid skill list for prompt: {prompt!r}")
                if not all(isinstance(skill, str) for skill in skills):
                    raise ValueError(f"invalid skill value for prompt: {prompt!r}")
                if prompt in records:
                    raise ValueError(f"duplicate benchmark prompt: {prompt!r}")
                records[prompt] = record
                order.append(prompt)
            if not records:
                raise ValueError("benchmark contains no prompts")
        except Exception as exc:
            logger.warning("GenEval2 benchmark loading failed from %s: %s", path, exc)
            return {}, [], False
        self._benchmark_digest = hashlib.sha256(normalized).hexdigest()
        return records, order, True

    def _is_canonical_benchmark(self) -> bool:
        return (
            len(self._benchmark) == BENCHMARK_PROMPTS
            and getattr(self, "_benchmark_digest", None) == BENCHMARK_SHA256
        )

    def setup(self) -> None:
        if self.test_mode:
            return
        if not self._benchmark_valid:
            self._backend = "unavailable"
            logger.warning("GenEval2 setup skipped because benchmark data is invalid")
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            from ayase.config import download_hf_snapshot

            model_path = self.config.get("model_path")
            if model_path is None:
                model_path = download_hf_snapshot(
                    MODEL_ID, self.models_dir, revision=MODEL_REVISION
                )
            device_map = self.config.get("device_map", "auto")
            device = self.config.get("device", "auto")
            if device_map == "auto" and device != "auto":
                device_map = device
            dtype = self.config.get("dtype", "auto")
            self._processor = AutoProcessor.from_pretrained(str(model_path))
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_path), dtype=dtype, device_map=device_map
            )
            self._model.eval()
            self._torch = torch
            self._backend = "qwen3_vl"
        except Exception as exc:
            self._processor = None
            self._model = None
            self._torch = None
            self._backend = "unavailable"
            logger.warning("GenEval2 setup failed: %s", exc)

    @staticmethod
    def _caption(sample: Sample) -> Optional[str]:
        return sample.caption.text if sample.caption is not None else None

    @staticmethod
    def _validate_vqa_list(value: Any, *, custom: bool) -> Optional[List[List[str]]]:
        if not isinstance(value, list) or not value:
            return None
        result: List[List[str]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                return None
            question, answer = item
            if not isinstance(question, str) or not isinstance(answer, str):
                return None
            if custom and not question.startswith("How many") and answer != "Yes":
                return None
            result.append([question, answer])
        return result

    def _questions_for(self, sample: Sample) -> Optional[Tuple[str, Dict[str, Any], bool]]:
        caption = self._caption(sample)
        if "geneval2_vqa_list" in sample.metadata:
            custom = sample.metadata.get("geneval2_vqa_list")
            vqa_list = self._validate_vqa_list(custom, custom=True)
            if vqa_list is None:
                logger.warning("GenEval2 custom VQA list is invalid for %s", sample.path)
                return None
            return caption or "", {
                "prompt": caption or "",
                "vqa_list": vqa_list,
                "atom_count": sample.metadata.get("geneval2_atom_count", len(vqa_list)),
                "skills": sample.metadata.get("geneval2_skills", []),
            }, True
        if caption is not None and caption in self._benchmark:
            return caption, self._benchmark[caption], not self._canonical_benchmark
        return None

    def process(self, sample: Sample) -> Sample:
        if sample.is_video or self._backend != "qwen3_vl":
            return sample
        selected = self._questions_for(sample)
        if selected is None:
            return sample
        prompt, record, custom = selected
        vqa_list = self._validate_vqa_list(record.get("vqa_list"), custom=custom)
        if vqa_list is None:
            return sample
        try:
            probabilities = [
                score_question(
                    self._processor, self._model, self._torch, str(sample.path), question, answer
                )
                for question, answer in vqa_list
            ]
            if not all(math.isfinite(value) and value >= 0 for value in probabilities):
                return sample
            am_score, gm_score = aggregate_atom_scores(probabilities)
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.soft_tifa_am_score = am_score
            sample.quality_metrics.soft_tifa_gm_score = gm_score
            sample.metadata["geneval2"] = {
                "prompt": prompt,
                "identity": "custom" if custom else "benchmark",
                "matched_benchmark": not custom,
                "atom_count": record.get("atom_count"),
                "skills": record.get("skills", []),
                "vqa_list": vqa_list,
                "atom_probabilities": probabilities,
            }
        except Exception as exc:
            logger.warning("GenEval2 failed for %s: %s", sample.path, exc)
        return sample

    def post_process(self, all_samples: Sequence[Sample]) -> None:
        if (
            not hasattr(self, "pipeline")
            or self.pipeline is None
            or not self._canonical_benchmark
        ):
            return
        self.pipeline.add_dataset_metric("geneval2_am_score", None)
        self.pipeline.add_dataset_metric("geneval2_gm_score", None)
        successful: Dict[str, List[Tuple[float, float]]] = {}
        official_inputs: Dict[str, int] = {}
        has_custom = False
        for sample in all_samples:
            if sample.is_video:
                continue
            caption = self._caption(sample)
            if caption in self._benchmark:
                official_inputs[caption] = official_inputs.get(caption, 0) + 1
            if "geneval2_vqa_list" in sample.metadata:
                has_custom = True
            metadata = sample.metadata.get("geneval2")
            if not isinstance(metadata, dict):
                continue
            if metadata.get("identity") == "custom":
                has_custom = True
                continue
            prompt = metadata.get("prompt")
            metrics = sample.quality_metrics
            if (
                prompt in self._benchmark
                and metrics is not None
                and metrics.soft_tifa_am_score is not None
                and metrics.soft_tifa_gm_score is not None
                and math.isfinite(metrics.soft_tifa_am_score)
                and math.isfinite(metrics.soft_tifa_gm_score)
                and metrics.soft_tifa_am_score >= 0
                and metrics.soft_tifa_gm_score >= 0
            ):
                successful.setdefault(prompt, []).append(
                    (metrics.soft_tifa_am_score, metrics.soft_tifa_gm_score)
                )
        benchmark_size = len(self._benchmark)
        coverage = len(successful) / benchmark_size if benchmark_size else 0.0
        self.pipeline.add_dataset_metric("geneval2_coverage", coverage)
        complete = (
            benchmark_size > 0
            and not has_custom
            and len(successful) == benchmark_size
            and all(len(values) == 1 for values in successful.values())
            and len(official_inputs) == benchmark_size
            and all(count == 1 for count in official_inputs.values())
        )
        if not complete:
            return
        ordered = [successful[prompt][0] for prompt in self._benchmark_order]
        self.pipeline.add_dataset_metric(
            "geneval2_am_score", 100.0 * sum(value[0] for value in ordered) / benchmark_size
        )
        self.pipeline.add_dataset_metric(
            "geneval2_gm_score", 100.0 * sum(value[1] for value in ordered) / benchmark_size
        )

    def on_dispose(self) -> None:
        self._processor = None
        self._model = None
        self._torch = None
        self._backend = None
        super().on_dispose()

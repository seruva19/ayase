"""Score instruction-guided image edits with the published EditScore protocol.

``sample.reference_path`` is the source image, ``sample.path`` is the edited
image, and the caption (or edited-image sidecar) is the edit instruction.
The pinned Qwen3-VL-4B base and EditScore LoRA make two sampled generations per
pass: source+edited for prompt following/consistency and edited-only for
perceptual quality. Scores are 0--10 (higher is better). Invalid, refused, or
fabricated parser-fallback responses leave all metrics unset.
"""

import gc
import logging
import math
from pathlib import Path

from ayase.config import download_hf_snapshot
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

SOURCE_REVISION = "4609c5d2ebb62fdebf665d3c924686d896ef1f74"
BASE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
BASE_MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
ADAPTER_ID = "EditScore/EditScore-Qwen3-VL-4B-Instruct"
ADAPTER_REVISION = "87dcfd0eba27335c036ef9616fa0e725141ba571"


class EditScoreModule(PipelineModule):
    """Published EditScore Qwen3-VL-4B image-edit evaluator."""

    name = "edit_score"
    implementation_status = "implemented"
    implementation_notes = "Pinned author snapshots and bundled EditScore implementation are instantiated and invoked on image-edit/reference/instruction inputs; validates/maps native output axes. No CLIP-only edit proxy."
    implementation_review = "Source inspection; confidence high; pretrained inference not validated."
    description = "Published EditScore instruction-guided image-edit evaluator"
    input_type = "image"
    provenance = "published"
    sources = {
        field: "EditScore — https://github.com/VectorSpaceLab/EditScore/tree/" + SOURCE_REVISION
        for field in (
            "edit_score_overall",
            "edit_score_prompt_following",
            "edit_score_consistency",
            "edit_score_perceptual_quality",
        )
    }
    implementation_sources = (
        "../third_party/edit_score/__init__.py",
        "../third_party/edit_score/qwen3vl.py",
        "../third_party/edit_score/utils.py",
        "../third_party/edit_score/_upstream_utils.py",
        "../third_party/edit_score/vie_prompts.py",
        "../third_party/edit_score/LICENSE",
        "../third_party/edit_score/NOTICE",
    )
    models = [
        {
            "id": BASE_MODEL_ID,
            "type": "huggingface",
            "revision": BASE_MODEL_REVISION,
            "task": "Qwen3-VL-4B base for EditScore",
            "auto_download": True,
            "notes": "Apache-2.0; pinned local snapshot",
        },
        {
            "id": ADAPTER_ID,
            "type": "huggingface",
            "revision": ADAPTER_REVISION,
            "task": "EditScore Qwen3-VL-4B LoRA adapter",
            "auto_download": True,
            "notes": (
                "Hugging Face metadata says license:other; no standalone adapter "
                "license was identified; terms require clarification"
            ),
        },
    ]
    metric_info = {
        "edit_score_overall": "EditScore geometric SC/PQ score (0-10, higher=better)",
        "edit_score_prompt_following": "Edit instruction-following score (0-10)",
        "edit_score_consistency": "Minimal-effective-edit consistency score (0-10)",
        "edit_score_perceptual_quality": "Minimum naturalness/artifact score (0-10)",
    }
    deviations = {
        field: (
            "Author evaluate body, prompts, preprocessing, generation, and aggregation are "
            "preserved. A parser gate rejects the author's random give-up scores, invalid raw "
            "score arrays, and last-resort zero padding; failures remain missing."
        )
        for field in sources
    }
    metric_groups = {
        "edit_score_overall": "alignment",
        "edit_score_prompt_following": "alignment",
        "edit_score_consistency": "alignment",
        "edit_score_perceptual_quality": "nr_quality",
    }
    default_config = {
        "model_path": None,
        "adapter_path": None,
        "models_dir": "models",
        "auto_download": True,
        "device": "auto",
        "temperature": 0.7,
        "num_pass": 1,
        "reduction": "average_last",
        "seed": 42,
        "score_range": 25,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._scorer = None
        self._backend = None
        self._ml_available = False
        self.runtime_trace = {
            "source_revision": SOURCE_REVISION,
            "base_model": f"{BASE_MODEL_ID}@{BASE_MODEL_REVISION}",
            "adapter": f"{ADAPTER_ID}@{ADAPTER_REVISION}",
            "protocol": "two-images SC; edited-only PQ",
            "score_range": int(self.config.get("score_range", 25)),
            "temperature": float(self.config.get("temperature", 0.7)),
            "num_pass": int(self.config.get("num_pass", 1)),
            "reduction": str(self.config.get("reduction", "average_last")),
            "seed": int(self.config.get("seed", 42)),
        }

    def _resolve_snapshot(self, key, repo_id, revision):
        override = self.config.get(key)
        if override:
            path = Path(override).expanduser().resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"EditScore {key} is not a directory: {path}")
            return path
        if not self.config.get("auto_download", True):
            raise FileNotFoundError(f"EditScore {repo_id} is not provisioned")
        return download_hf_snapshot(
            repo_id,
            str(self.config.get("models_dir", "models")),
            revision=revision,
            allow_patterns=(
                ["adapter_config.json", "adapter_model.safetensors", "README.md", "LICENSE*"]
                if key == "adapter_path"
                else ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.py", "LICENSE*"]
            ),
            ignore_patterns=["*.bin", "trainer_state.json"],
        )

    def setup(self):
        if self.test_mode:
            return
        self._scorer = None
        self._backend = None
        self._ml_available = False
        try:
            import torch

            device = str(self.config.get("device", "auto"))
            if device not in {"auto", "cuda"}:
                raise ValueError("published EditScore supports only its CUDA route")
            if not torch.cuda.is_available():
                raise RuntimeError("published EditScore requires CUDA")
            base = self._resolve_snapshot(
                "model_path", BASE_MODEL_ID, BASE_MODEL_REVISION
            )
            adapter = self._resolve_snapshot(
                "adapter_path", ADAPTER_ID, ADAPTER_REVISION
            )
            from ayase.third_party.edit_score import EditScore

            self._scorer = EditScore(
                backbone="qwen3vl",
                model_name_or_path=str(base),
                lora_path=str(adapter),
                score_range=self.runtime_trace["score_range"],
                temperature=self.runtime_trace["temperature"],
                num_pass=self.runtime_trace["num_pass"],
                reduction=self.runtime_trace["reduction"],
                seed=self.runtime_trace["seed"],
            )
            self._backend = (
                f"editscore:qwen3vl4b@{BASE_MODEL_REVISION}+lora@{ADAPTER_REVISION}"
            )
            self._ml_available = True
        except Exception as exc:
            self._scorer = None
            self._backend = "unavailable"
            logger.warning("EditScore unavailable: %s", exc)

    @staticmethod
    def _instruction(sample):
        if sample.caption is not None and sample.caption.text.strip():
            return sample.caption.text.strip()
        sidecar = sample.path.with_suffix(".txt")
        try:
            return sidecar.read_text(encoding="utf-8").strip() if sidecar.is_file() else None
        except (OSError, UnicodeError):
            return None

    def process(self, sample):
        if (
            sample.is_video
            or not self._ml_available
            or self._scorer is None
            or sample.reference_path is None
        ):
            return sample
        instruction = self._instruction(sample)
        if not instruction:
            return sample
        try:
            from PIL import Image

            with Image.open(sample.reference_path) as source_file:
                source = source_file.convert("RGB")
            with Image.open(sample.path) as edited_file:
                edited = edited_file.convert("RGB")
            result = self._scorer.evaluate([source, edited], instruction)
            fields = {
                "edit_score_overall": result["overall"],
                "edit_score_prompt_following": result["prompt_following"],
                "edit_score_consistency": result["consistency"],
                "edit_score_perceptual_quality": result["perceptual_quality"],
            }
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 10
                for value in fields.values()
            ):
                raise ValueError("EditScore returned invalid public scores")
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            for name, value in fields.items():
                setattr(sample.quality_metrics, name, float(value))
            sample.quality_metrics.metric_backends[self.name] = self._backend
            sample.metadata[self.name] = {
                **self.runtime_trace,
                "backend": self._backend,
                "reasoning": {
                    "SC": result["SC_reasoning"],
                    "PQ": result["PQ_reasoning"],
                },
                "expert_override": bool(
                    self.config.get("model_path") or self.config.get("adapter_path")
                ),
            }
        except Exception as exc:
            logger.warning("EditScore failed for %s: %s", sample.path, exc)
        return sample

    def on_dispose(self):
        self._scorer = None
        self._ml_available = False
        self._backend = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        super().on_dispose()


__all__ = ["EditScoreModule"]

"""Published DeQA-Score Mix3 no-reference image quality rating.

Uses the authors' exact ``model.score`` image path: RGB conversion, square
padding, 448-pixel CLIP preprocessing, the quality prompt, and the weighted
five-level softmax. Scores are on the native 1--5 scale (higher is better).
The pinned legacy Transformers runtime is isolated because its remote model
code globally patches Llama classes. CUDA float16 with eager attention is the
only supported execution path; unavailable backends leave the score unset.
"""

import logging
import math
import os
from pathlib import Path

from ayase.config import download_hf_snapshot
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


SOURCE_REVISION = "4342013af7fa5f450bb3a0cc8e135a61928d50bf"
MODEL_ID = "zhiyuanyou/DeQA-Score-Mix3"
MODEL_REVISION = "246e219b7e3bc9b5dff5269c1a8564087aefca3f"
LEVEL_IDS = (15129, 1781, 6534, 6460, 4319)

SNAPSHOT_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "configuration_mplug_owl2.py",
    "modeling_mplug_owl2_huggingface.py",
    "modeling_llama2.py",
    "modeling_attn_mask_utils.py",
    "visual_encoder.py",
    "__init__.py",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "README.md",
)


class DeQAScoreModule(PipelineModule):
    """Official full-tuned DeQA-Score Mix3 image scorer."""

    name = "deqa_score"
    implementation_status = "implemented"
    implementation_notes = "Pinned official Mix3 snapshot and remote model class, verified level-token IDs, CUDA float16 native model.score quality path and native 1-5 score validation."
    implementation_review = "Source inspection; confidence high; pretrained inference not validated."
    description = "Official DeQA-Score Mix3 native image-quality rating"
    input_type = "image"
    provenance = "published"
    sources = {
        "deqa_score": (
            "You et al., DeQA-Score; https://github.com/zhiyuanyou/DeQA-Score "
            f"at {SOURCE_REVISION}"
        )
    }
    implementation_sources = (
        "../third_party/deqa_score/LICENSE",
        "../third_party/deqa_score/NOTICE",
    )
    models = [
        {
            "id": MODEL_ID,
            "type": "huggingface",
            "revision": MODEL_REVISION,
            "task": "No-reference image quality assessment",
            "size": "16.4 GB parameters (four safetensors shards)",
            "vram": "CUDA model; approximately 16.4 GB of weights plus activations",
            "auto_download": True,
            "notes": (
                f"Full Mix3 revision {MODEL_REVISION}; source revision {SOURCE_REVISION}; "
                "native Transformers 4.36.1, eager attention, float16; MIT model/source "
                "with Apache-2.0 notices retained for derived mPLUG-Owl/LLaVA code"
            ),
        }
    ]
    metric_info = {
        "deqa_score": "Native DeQA-Score Mix3 image-quality rating (1-5, higher=better)"
    }
    metric_groups = {"deqa_score": "nr_quality"}
    default_config = {
        "model_path": None,
        "models_dir": "models",
        "auto_download": True,
        "device": "auto",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_path = self.config.get("model_path")
        self.models_dir = self.config.get("models_dir", "models")
        self.auto_download = bool(self.config.get("auto_download", True))
        self.device = str(self.config.get("device", "auto"))
        self._model = None
        self._backend = None
        self._ml_available = False
        self.runtime_trace = {
            "source_revision": SOURCE_REVISION,
            "model_revision": MODEL_REVISION,
            "input": "image-only RGB",
            "precision": "float16",
            "attention": "eager",
        }

    @staticmethod
    def _validate_snapshot(root: Path) -> None:
        missing = [name for name in SNAPSHOT_FILES if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"DeQA-Score snapshot is incomplete under {root}: missing {', '.join(missing)}"
            )

    def _resolve_snapshot(self) -> Path:
        if self.model_path:
            root = Path(self.model_path).expanduser().resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"DeQA-Score model_path is not a directory: {root}")
        else:
            if not self.auto_download:
                raise FileNotFoundError(
                    "DeQA-Score is not provisioned and auto_download is disabled"
                )
            root = download_hf_snapshot(
                MODEL_ID,
                self.models_dir,
                revision=MODEL_REVISION,
                allow_patterns=list(SNAPSHOT_FILES),
                ignore_patterns=["*.bin", "trainer_state.json"],
            )
        self._validate_snapshot(root)
        return root

    def setup(self) -> None:
        if self.test_mode:
            return
        self._model = None
        self._backend = None
        self._ml_available = False
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM

            if self.device not in {"auto", "cuda"} and not self.device.startswith("cuda:"):
                raise ValueError("official DeQA-Score inference requires a CUDA device")
            if not torch.cuda.is_available():
                raise RuntimeError("official DeQA-Score inference requires CUDA")

            snapshot = self._resolve_snapshot()
            previous_offline = {
                name: os.environ.get(name)
                for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
            }
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                model_config = AutoConfig.from_pretrained(
                    str(snapshot),
                    trust_remote_code=True,
                    local_files_only=True,
                )
                # The pinned remote constructor resolves its tokenizer and CLIP
                # processor from this value without forwarding local_files_only.
                model_config._name_or_path = str(snapshot)
                model = AutoModelForCausalLM.from_pretrained(
                    str(snapshot),
                    config=model_config,
                    trust_remote_code=True,
                    local_files_only=True,
                    use_safetensors=True,
                    attn_implementation="eager",
                    torch_dtype=torch.float16,
                    device_map=("auto" if self.device == "auto" else {"": self.device}),
                )
            finally:
                for name, value in previous_offline.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

            configured_ids = tuple(int(value) for value in model.config.level_ids)
            tokenized_ids = tuple(int(value) for value in model.preferential_ids_)
            if configured_ids != LEVEL_IDS or tokenized_ids != LEVEL_IDS:
                raise RuntimeError(
                    "DeQA-Score level-token IDs do not match the pinned Mix3 protocol"
                )
            self._model = model.eval()
            self._backend = f"hf:{MODEL_ID}@{MODEL_REVISION}:eager-float16-image"
            self._ml_available = True
        except Exception as exc:
            self._model = None
            self._backend = "unavailable"
            self._ml_available = False
            logger.warning("DeQA-Score unavailable: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if sample.is_video or not self._ml_available or self._model is None:
            return sample
        try:
            from PIL import Image

            with Image.open(sample.path) as opened:
                image = opened.convert("RGB")
                raw = self._model.score([image], task_="quality", input_="image")
            if int(raw.numel()) != 1:
                raise ValueError(
                    f"DeQA-Score returned {int(raw.numel())} values for one image"
                )
            score = float(raw.detach().float().cpu().reshape(-1)[0].item())
            if not math.isfinite(score) or not 1.0 <= score <= 5.0:
                raise ValueError(f"DeQA-Score returned invalid native score {score!r}")
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.deqa_score = score
            sample.quality_metrics.metric_backends["deqa_score"] = self._backend
            sample.metadata["deqa_score"] = {
                **self.runtime_trace,
                "device": str(getattr(self._model, "device", self.device)),
                "requested_device": self.device,
                "dtype": "float16",
                "backend": self._backend,
                "expert_override": bool(self.model_path),
            }
        except Exception as exc:
            logger.warning("DeQA-Score failed for %s: %s", sample.path, exc)
        return sample

    def on_dispose(self) -> None:
        self._model = None
        self._ml_available = False
        self._backend = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        super().on_dispose()


__all__ = ["DeQAScoreModule"]

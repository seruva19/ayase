"""Human-aligned quality assessment for instruction-guided video editing.

VE-Bench (AAAI 2025) combines text/edit alignment, source-edited temporal
correlation, and DOVER-derived visual quality. The upstream evaluator returns
one comparative scalar score (higher is better); it is not an absolute MOS.

The implementation uses the authors' MIT-licensed ``vebench`` package and its
six upstream checkpoints. A CUDA GPU is required by the upstream evaluator.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from ayase.config import download_hf_snapshot
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

from ._reward_utils import get_prompt

logger = logging.getLogger(__name__)


_MODEL_REPO = "AkaneTendo25/ayase-runtime-assets"
_MODEL_REVISION = "377ac94f1fadca1e35b7171c7860ccad6ae9fd1a"
_WEIGHTS: Tuple[Tuple[str, int, str], ...] = (
    (
        "convnext_tiny_1k_224_ema.pth",
        114_414_741,
        "14f3164e3ea6ac32ab3f574f528ce817696c9176fad4221e0a77a905a7360595",
    ),
    (
        "e-bench-blip_head_videoQA_9_eval_s_finetuned.pth",
        162_355_787,
        "9762e485e092b93045b6cfab0573181130f28cd5acb946c7335a73b988610259",
    ),
    (
        "e-bench-dover_head_videoQA_0_eval_n_finetuned.pth",
        259_082_391,
        "7dfd43355f3b8f74942b1be078cf4c530ea42286bb90407e59aae8933ea0085a",
    ),
    (
        "e-bench-uniformer-src-edit_head_videoQA_3_eval_s_finetuned.pth",
        764_601_409,
        "043729d2a13ce8b299cf9aebed92ae0666edf7d32ed06a799befe8c37f2c98b2",
    ),
    (
        "k400+k710_uniformerv2_b16_8x224.pth",
        458_289_355,
        "743da61c97f6281bd11ef1d364c2101698c8e89ff7722ee957378e36ef344005",
    ),
    (
        "model_large.pth",
        3_894_199_750,
        "f39dab16712aa8d3ed4acb774f0a691d7ddaa7ef6d9c02d395faa852776bb734",
    ),
)

_CWD_LOCK = threading.Lock()


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """Temporarily satisfy the upstream evaluator's relative checkpoint paths."""
    with _CWD_LOCK:
        previous = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)


class VEBenchModule(PipelineModule):
    """Score an edited video against its source video and edit instruction."""

    name = "vebench"
    description = "VE-Bench human-aligned instruction-guided video-edit quality (AAAI 2025)"
    default_config = {
        "models_dir": "models",
        "instruction": None,
        "warning_threshold": None,
    }
    required_packages = ["torch", "transformers", "vebench", "huggingface_hub"]
    models = [
        {
            "id": "vebench==1.0.0",
            "type": "pip_package",
            "install": "pip install 'vebench==1.0.0'",
            "task": "VE-Bench evaluator implementation",
            "auto_download": False,
            "notes": "MIT; AAAI 2025; CUDA required",
        },
        {
            "id": _MODEL_REPO,
            "type": "huggingface",
            "task": "Six VE-Bench evaluator checkpoints",
            "size": "5.65 GB",
            "vram": "~6 GB total evaluator peak",
            "auto_download": True,
            "notes": (
                f"Pinned revision {_MODEL_REVISION}; SHA-256-verified files under "
                "vebench/ckpts/"
            ),
        },
    ]
    metric_info = {
        "vebench_score": (
            "VE-Bench comparative video-edit quality combining instruction alignment, "
            "source-edit dynamics, and visual quality (higher=better)"
        )
    }
    metric_groups = {"vebench_score": "alignment"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._backend: Optional[str] = None
        self._model: Any = None
        self._torch: Any = None
        self._model_root: Optional[Path] = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("VE-Bench requires a CUDA GPU")
                return

            self._torch = torch
            snapshot_root = download_hf_snapshot(
                _MODEL_REPO,
                str(self.config.get("models_dir", "models")),
                revision=_MODEL_REVISION,
                allow_patterns=["vebench/ckpts/*.pth"],
            )
            self._model_root = snapshot_root / "vebench"
            checkpoint_dir = self._model_root / "ckpts"
            self._ensure_weights(checkpoint_dir)

            with _working_directory(self._model_root):
                original_torch_load = self._install_compatibility_shims(torch)
                try:
                    from vebench import VEBenchModel

                    self._model = VEBenchModel()
                finally:
                    torch.load = original_torch_load
            self._backend = "vebench"
        except Exception as exc:
            self._backend = None
            self._model = None
            logger.warning("VE-Bench setup failed: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or self._backend != "vebench":
            return sample
        if self._model is None or sample.reference_path is None:
            return sample

        instruction = get_prompt(sample, self.config, key="instruction")
        if not instruction:
            return sample

        try:
            score = self._model.evaluate(
                instruction,
                str(Path(sample.reference_path).resolve()),
                str(sample.path.resolve()),
            )
            if hasattr(score, "item"):
                score = score.item()
            value = float(score)
            if not math.isfinite(value):
                logger.warning("VE-Bench returned a non-finite score for %s", sample.path)
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.vebench_score = value

            threshold = self.config.get("warning_threshold")
            if threshold is not None and value < float(threshold):
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low VE-Bench video-edit quality: {value:.3f}",
                        details={"vebench_score": value},
                    )
                )
        except Exception as exc:
            logger.warning("VE-Bench failed for %s: %s", sample.path, exc)
        return sample

    def on_dispose(self) -> None:
        self._model = None
        self._backend = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        super().on_dispose()

    @staticmethod
    def _install_compatibility_shims(torch: Any) -> Any:
        """Bridge upstream code to Transformers>=4.45 and PyTorch>=2.6."""
        import transformers.modeling_utils as modeling_utils
        from transformers.pytorch_utils import (
            apply_chunking_to_forward,
            find_pruneable_heads_and_indices,
            prune_linear_layer,
        )

        modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
        modeling_utils.find_pruneable_heads_and_indices = (
            find_pruneable_heads_and_indices
        )
        modeling_utils.prune_linear_layer = prune_linear_layer

        original_load = torch.load

        def compatible_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        compatible_load._ayase_vebench_compat = True
        torch.load = compatible_load
        return original_load

    @staticmethod
    def _ensure_weights(checkpoint_dir: Path) -> None:
        for filename, expected_size, expected_sha256 in _WEIGHTS:
            destination = checkpoint_dir / filename
            if not destination.is_file():
                raise RuntimeError(f"Missing VE-Bench weight in Hugging Face snapshot: {filename}")
            actual_size = destination.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"VE-Bench weight {filename} has size {actual_size}, "
                    f"expected {expected_size}"
                )
            digest = hashlib.sha256()
            with destination.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise RuntimeError(f"VE-Bench weight {filename} failed SHA-256 verification")

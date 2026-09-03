"""Fine-grained video preference scoring with MJ-Video.

Downloads the public MJ-VIDEO-2B checkpoint during setup and runs the reference
reward architecture, vendored in-tree (``ayase.vendor.mj_video``) at the pinned
commit, in-process. Ayase downloads weights, never code. Exposes the learned overall reward and five
aspect rewards; all 28 criterion rewards are retained in sample metadata.
Higher rewards indicate stronger learned preference.
"""

import logging
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

#: Upstream revision the vendored runtime in ``ayase.vendor.mj_video`` was taken from.
SOURCE_REVISION = "cc1d2c9587a620e9ebd3599ae4cdd21b5fd7c87a"
SOURCE_HOME = "https://github.com/aiming-lab/MJ-Video"
TOKENIZER_BASE_URL = (
    "https://huggingface.co/internlm/internlm2-chat-1_8b/resolve"
)
ASPECT_FIELDS = (
    "mj_video_alignment_score",
    "mj_video_safety_score",
    "mj_video_fineness_score",
    "mj_video_coherence_score",
    "mj_video_fairness_score",
)
ASPECT_TO_CRITERIA = {
    0: [0, 1, 2, 3, 4],
    1: [5, 6, 7, 8, 9, 10],
    2: [11, 12, 13, 14, 15],
    3: [16, 17, 18, 19, 20, 21, 22],
    4: [23, 24, 25, 26, 27],
}


class MJVideoModule(PipelineModule):
    """Score each text-video pair with the upstream MJ-VIDEO-2B reward model."""

    name = "mj_video"
    description = "MJ-Video overall reward and five fine-grained preference aspects"
    default_config = {
        "model_name": "MJ-Bench/MJ-VIDEO-2B",
        "model_revision": None,
        "models_dir": "models",
        "tokenizer_base_url": TOKENIZER_BASE_URL,
        "tokenizer_revision": "main",
        "num_segments": 8,
        "max_new_tokens": 1024,
        "do_sample": True,
        "gating_temperature": 1.0,
        "gating_hidden_dim": 1024,
        "gating_n_hidden": 3,
    }
    required_packages = ["torch", "transformers", "safetensors", "decord"]
    models = [
        {
            "id": "MJ-Bench/MJ-VIDEO-2B",
            "type": "huggingface",
            "task": "Fine-grained video preference reward model",
            "size": "4.43 GB inference checkpoint",
            "auto_download": True,
            "notes": "Eight uniformly sampled frames by default",
        },
        {
            "id": "internlm/internlm2-chat-1_8b",
            "type": "huggingface",
            "task": "InternLM2 tokenizer code and SentencePiece model",
            "auto_download": True,
        },
    ]
    metric_info = {
        "mj_video_overall_score": "MJ-Video learned overall preference reward (higher=better)",
        "mj_video_alignment_score": "MJ-Video alignment aspect reward (higher=better)",
        "mj_video_safety_score": "MJ-Video safety aspect reward (higher=better)",
        "mj_video_fineness_score": "MJ-Video fineness aspect reward (higher=better)",
        "mj_video_coherence_score": "MJ-Video coherence/consistency reward (higher=better)",
        "mj_video_fairness_score": "MJ-Video bias/fairness aspect reward (higher=better)",
    }
    metric_groups = {
        "mj_video_overall_score": "alignment",
        "mj_video_alignment_score": "alignment",
        "mj_video_safety_score": "safety",
        "mj_video_fineness_score": "nr_quality",
        "mj_video_coherence_score": "temporal",
        "mj_video_fairness_score": "safety",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._backend: Optional[str] = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._load_video: Any = None
        self._prepare_chat_input: Any = None
        self._reward_config: Any = None
        self._device: Any = None
        self._dtype: Any = None

    @staticmethod
    def _vendored_source() -> Path:
        """Path of the in-tree MJ-Video runtime.

        Returns:
            Path: Directory holding the ``scripts`` tree the reward model is
            imported from.

        Raises:
            RuntimeError: The vendored tree is missing.
        """
        from ayase import vendor

        source_root = Path(vendor.__file__).resolve().parent / "mj_video"
        if not (source_root / "scripts" / "model").is_dir():
            raise RuntimeError("MJ-Video runtime is missing from ayase.vendor")
        return source_root

    @staticmethod
    def _ensure_unused_s3_compatibility() -> None:
        """Satisfy MJ-Video's eager training-dataset boto3 import for local inference."""
        try:
            import boto3  # type: ignore[import-not-found]  # noqa: F401

            return
        except ImportError:
            pass

        class UnavailableSession:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("S3 access is unavailable in MJ-Video local inference")

        package = types.ModuleType("boto3")
        package.__path__ = []  # type: ignore[attr-defined]
        session = types.ModuleType("boto3.session")
        session.Session = UnavailableSession  # type: ignore[attr-defined]
        package.session = session  # type: ignore[attr-defined]
        sys.modules.setdefault("boto3", package)
        sys.modules.setdefault("boto3.session", session)

    @staticmethod
    def _ensure_transformers_doc_compatibility() -> None:
        """Restore a removed documentation-only constant used by MJ-Video."""
        from transformers.models.llama import modeling_llama

        if not hasattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING"):
            modeling_llama.LLAMA_INPUTS_DOCSTRING = ""

    def _prepare_tokenizer_files(self, checkpoint: Path, models_dir: str) -> None:
        from ayase.config import download_model_file

        base_url = str(self.config.get("tokenizer_base_url", TOKENIZER_BASE_URL)).rstrip("/")
        revision = str(self.config.get("tokenizer_revision", "main"))
        for filename in (
            "tokenization_internlm2.py",
            "tokenization_internlm2_fast.py",
            "tokenizer.model",
        ):
            source = download_model_file(
                f"mj_video/tokenizer/{filename}",
                f"{base_url}/{revision}/{filename}",
                models_dir,
            )
            target = checkpoint / filename
            if target.exists():
                continue
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)

    @staticmethod
    def _configure_internvl_single_process() -> None:
        """Make InternVL's logging-only get_rank call safe outside distributed runs."""
        from internvl2 import modeling_internvl_chat

        if getattr(modeling_internvl_chat, "_ayase_single_process", False):
            return
        real_torch = modeling_internvl_chat.torch

        class DistributedProxy:
            def __getattr__(self, name: str) -> Any:
                return getattr(real_torch.distributed, name)

            @staticmethod
            def get_rank(*args: Any, **kwargs: Any) -> int:
                if real_torch.distributed.is_initialized():
                    return int(real_torch.distributed.get_rank(*args, **kwargs))
                return 0

        class TorchProxy:
            distributed = DistributedProxy()

            def __getattr__(self, name: str) -> Any:
                return getattr(real_torch, name)

        modeling_internvl_chat.torch = TorchProxy()
        modeling_internvl_chat._ayase_single_process = True

    #: Vendored components whose licence differs from Ayase's own.
    vendor_components = ('mj_video',)

    def setup(self) -> None:
        from ayase.licenses import announce

        announce(self.vendor_components)
        try:
            from ayase.config import download_hf_snapshot, download_model_file
            from ayase.runtime import resolve_torch_device

            models_dir = str(self.config.get("models_dir", "models"))
            checkpoint = download_hf_snapshot(
                str(self.config.get("model_name", "MJ-Bench/MJ-VIDEO-2B")),
                models_dir,
                revision=self.config.get("model_revision"),
                ignore_patterns=["optimizer.pt", "scheduler.pt", "training_args.bin"],
            )
            source_root = self._vendored_source()
            self._prepare_tokenizer_files(checkpoint, models_dir)
            scripts_dir = str((source_root / "scripts").resolve())
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            model_dir = str((source_root / "scripts" / "model").resolve())
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)

            self._ensure_unused_s3_compatibility()
            self._ensure_transformers_doc_compatibility()
            import torch
            from data_processor import load_video
            from model import (
                InternVLChatRewardModeling,
                InternVLChatRewardModelingConfig,
                prepare_chat_input,
            )
            from internvl2.configuration_internvl_chat import InternVLChatConfig
            from safetensors.torch import load_file
            from transformers import AutoTokenizer

            InternVLChatRewardModelingConfig.has_no_defaults_at_init = True
            InternVLChatConfig.has_no_defaults_at_init = True
            self._configure_internvl_single_process()
            self._device = torch.device(resolve_torch_device(self.config.get("device", "auto")))
            self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(
                str(checkpoint), trust_remote_code=True, use_fast=False
            )
            reward_config = InternVLChatRewardModelingConfig.from_pretrained(
                str(checkpoint),
                pad_token_id=tokenizer.pad_token_id,
                num_objectives=28,
                num_aspects=5,
                aspect2criteria=ASPECT_TO_CRITERIA,
                gating_temperature=float(self.config.get("gating_temperature", 1.0)),
                gating_hidden_dim=int(self.config.get("gating_hidden_dim", 1024)),
                gating_n_hidden=int(self.config.get("gating_n_hidden", 3)),
            )
            model = InternVLChatRewardModeling(name=str(checkpoint), config=reward_config)
            weights = load_file(str(checkpoint / "model.safetensors"), device="cpu")
            model.load_state_dict(weights, strict=True)
            model.config.pad_token_id = tokenizer.pad_token_id
            model.model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            model = model.to(device=self._device, dtype=self._dtype).eval()

            self._torch = torch
            self._tokenizer = tokenizer
            self._reward_config = reward_config
            self._model = model
            self._load_video = load_video
            self._prepare_chat_input = prepare_chat_input
            self._backend = "mj_video"
        except Exception as exc:
            self._backend = "unavailable"
            logger.warning("MJ-Video unavailable: %s", exc)
            logger.debug("MJ-Video setup traceback", exc_info=True)

    @staticmethod
    def _caption(sample: Sample) -> str:
        if sample.caption is not None and sample.caption.text:
            return sample.caption.text
        sidecar = sample.path.with_suffix(".txt")
        try:
            return sidecar.read_text(encoding="utf-8").strip() if sidecar.is_file() else ""
        except OSError:
            return ""

    @staticmethod
    def _tensor_values(value: Any) -> List[float]:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().reshape(-1).tolist()
        if not isinstance(value, list):
            value = [value]
        return [float(item) for item in value]

    def _attach_output(self, sample: Sample, output: Any) -> None:
        overall = self._tensor_values(output.score)
        aspects = self._tensor_values(output.aspect_scores)
        criteria = self._tensor_values(output.rewards)
        if not overall or len(aspects) != 5 or len(criteria) != 28:
            raise ValueError("MJ-Video returned an unexpected reward shape")
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.mj_video_overall_score = overall[0]
        sample.quality_metrics.mj_video_alignment_score = aspects[0]
        sample.quality_metrics.mj_video_safety_score = aspects[1]
        sample.quality_metrics.mj_video_fineness_score = aspects[2]
        sample.quality_metrics.mj_video_coherence_score = aspects[3]
        sample.quality_metrics.mj_video_fairness_score = aspects[4]
        sample.metadata["mj_video_criteria_scores"] = criteria

    def process(self, sample: Sample) -> Sample:
        if self._backend != "mj_video" or not sample.is_video:
            return sample
        try:
            pixel_values, patch_counts = self._load_video(
                str(sample.path),
                num_segments=int(self.config.get("num_segments", 8)),
                max_num=1,
            )
            prefix = "".join(
                f"Frame{index + 1}: <image>\n" for index in range(len(patch_counts))
            )
            pixel_values = pixel_values.to(device=self._device, dtype=self._dtype)
            input_ids, attention_mask = self._prepare_chat_input(
                self._reward_config,
                self._tokenizer,
                pixel_values,
                prefix + self._caption(sample),
                {
                    "max_new_tokens": int(self.config.get("max_new_tokens", 1024)),
                    "do_sample": bool(self.config.get("do_sample", True)),
                },
                device=self._device,
            )
            with self._torch.inference_mode():
                output = self._model(pixel_values, input_ids, attention_mask)
            self._attach_output(sample, output)
        except Exception as exc:
            logger.warning("MJ-Video failed for %s: %s", sample.path, exc)
        return sample

    def on_dispose(self) -> None:
        self._model = None
        self._tokenizer = None
        self._torch = None
        super().on_dispose()
